"""Stage-1 bytes/param probe for full-FT CPU offload (scratch harness).

Engines:
  hf     backpropagate Trainer(mode="full", full_ft_offload=True) as in PR #222
         (accelerate FSDP2: fp32 upcast + torch AdamW).
  fsdp2  our own fully_shard + CPUOffloadPolicy, params kept in --dtype,
         a manual training loop, and a pluggable optimizer.

Prints one PROBE_RECEIPT json line (host bytes/param for params / grads /
optimizer state, peak RSS for the training phase, peak VRAM, s/step with a
fwd/bwd/opt split, and with --precision-check the % of parameters that
changed after optimizer step 1).

Optimizers (--optim): torch_adamw, torch_adafactor, sgd; af_nearest / af_sr /
af_kahan = factored Adafactor (no momentum) computed in fp32 and written back
to the param dtype with round-to-nearest / stochastic rounding / a Kahan
compensation buffer; adamw_sr = AdamW with bf16 moments + stochastic rounding.
--opt-device cuda streams each param + grad to the GPU for the step.

Example:
  python scripts/offload_probe.py --model HuggingFaceTB/SmolLM2-360M-Instruct       --engine fsdp2 --dtype bf16 --optim af_sr --opt-device cuda --steps 20 --precision-check

Stage-1 scratch harness for the "make 7B real" work (feat/offload-7b); not
part of the library or the test suite.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time

import psutil
import torch

GB = 1024 ** 3

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--engine", choices=["hf", "fsdp2"], required=True)
ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
ap.add_argument("--optim", default="adamw",
                help="torch_adamw | torch_adafactor | sgd | af_nearest | af_sr | af_kahan | adamw_sr")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--seq", type=int, default=256)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--precision-check", action="store_true")
ap.add_argument("--no-pin", action="store_true")
ap.add_argument("--opt-device", default="cpu")
ap.add_argument("--no-ckpt", action="store_true")
ap.add_argument("--tag", default="")
args = ap.parse_args()


class RSSMon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.proc = psutil.Process()
        self.peak = 0
        self.phase_peak = 0
        self.stop = False

    def run(self):
        while not self.stop:
            r = self.proc.memory_info().rss
            self.peak = max(self.peak, r)
            self.phase_peak = max(self.phase_peak, r)
            time.sleep(0.01)

    def reset_phase(self):
        self.phase_peak = self.proc.memory_info().rss


mon = RSSMon()
mon.start()

ROWS = [
    ("What is Python?", "Python is a high-level, readable programming language."),
    ("Explain recursion in one sentence.", "Recursion is when a function calls itself on a smaller input until a base case."),
    ("What does HTTP stand for?", "HyperText Transfer Protocol."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Name a primary color.", "Blue is a primary color."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("Define an algorithm briefly.", "An algorithm is a finite sequence of steps that solves a problem."),
    ("What is a variable?", "A named container that stores a value in a program."),
]

receipt: dict = {"tag": args.tag, "model": args.model, "engine": args.engine, "dtype": args.dtype,
                 "optim": args.optim, "opt_device": args.opt_device, "malloc": os.environ.get("LD_PRELOAD", "") or os.environ.get("MALLOC_ARENA_MAX", ""), "lr": args.lr, "steps": args.steps, "seq": args.seq, "batch": args.batch}

# --------------------------------------------------------------------------- hf
if args.engine == "hf":
    import tempfile

    import trl

    from backpropagate.trainer import Trainer

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "a.jsonl"), "w") as fh:
        for q, a in ROWS:
            fh.write(json.dumps({"messages": [{"role": "user", "content": q},
                                              {"role": "assistant", "content": a}]}) + "\n")
    stamps: list[float] = []
    orig = trl.SFTTrainer.training_step

    def spy(self, *a, **k):
        stamps.append(time.perf_counter())
        return orig(self, *a, **k)

    trl.SFTTrainer.training_step = spy
    t = Trainer(model=args.model, use_unsloth=False, mode="full", full_ft_offload=True,
                max_seq_length=args.seq, batch_size=args.batch, gradient_accumulation=1,
                learning_rate=args.lr, output_dir=os.path.join(d, "o"), report_to="none")
    run = t.train(os.path.join(d, "a.jsonl"), steps=args.steps)
    stamps.append(time.perf_counter())
    params = list(t._model.parameters())
    n = sum(p.numel() for p in params)
    receipt.update(num_params=n, final_loss=run.final_loss,
                   s_per_step=(stamps[-1] - stamps[1]) / max(1, len(stamps) - 2) if len(stamps) > 2 else None,
                   param_dtype=str(params[0].dtype))
    receipt["peak_vram_alloc_gb"] = round(torch.cuda.max_memory_allocated() / GB, 3)
    receipt["peak_rss_gb"] = round(mon.peak / GB, 3)
    print("PROBE_RECEIPT " + json.dumps(receipt), flush=True)
    raise SystemExit(0)

# ------------------------------------------------------------------------ fsdp2
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer

for k, v in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511", "RANK": "0", "WORLD_SIZE": "1",
             "LOCAL_RANK": "0"}.items():
    os.environ.setdefault(k, v)
dist.init_process_group("nccl", rank=0, world_size=1)
torch.cuda.set_device(0)
rss0 = psutil.Process().memory_info().rss
receipt["rss_after_cuda_init_gb"] = round(rss0 / GB, 3)

pdtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=pdtype)
model.config.use_cache = False
receipt["rss_after_load_gb"] = round(psutil.Process().memory_info().rss / GB, 3)
if not args.no_ckpt:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=pdtype)
off = CPUOffloadPolicy(pin_memory=not args.no_pin)
for layer in model.model.layers:
    fully_shard(layer, mp_policy=mp, offload_policy=off)
fully_shard(model, mp_policy=mp, offload_policy=off)
receipt["rss_after_shard_gb"] = round(psutil.Process().memory_info().rss / GB, 3)
params = [p for p in model.parameters() if p.requires_grad]
n = sum(p.numel() for p in params)
receipt["num_params"] = n
receipt["param_dtype"] = str(params[0].dtype)
receipt["param_device"] = str(params[0].to_local().device)


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def _round_into(p32: torch.Tensor, dst: torch.Tensor, mode: str, comp: torch.Tensor | None):
    """Write fp32 value p32 into bf16/fp32 dst with the chosen rounding."""
    if dst.dtype == torch.float32:
        dst.copy_(p32)
        return
    if mode == "sr":
        bits = p32.view(torch.int32)
        noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device)
        rounded = (bits + noise) & -65536  # clear low 16 bits
        dst.copy_(rounded.view(torch.float32))  # exact: low bits are zero
    elif mode == "kahan":
        new = p32 + comp.float()
        dst.copy_(new)
        comp.copy_(new - dst.float())
    else:
        dst.copy_(p32)


class LowMemOpt(torch.optim.Optimizer):
    """Adafactor (factored v, no momentum, update clip 1.0, absolute lr) or AdamW,
    computing in fp32 and writing back to the param dtype with nearest / stochastic
    rounding / Kahan compensation."""

    def __init__(self, params, lr, algo="adafactor", rounding="nearest", state_dtype=torch.float32, device="cpu"):
        super().__init__(params, {"lr": lr})
        self.algo, self.rounding, self.state_dtype = algo, rounding, state_dtype
        self.dev = torch.device(device)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                w_host = local(p)
                w = w_host.to(self.dev, non_blocking=True)
                g = local(p.grad).to(self.dev, non_blocking=True).float()
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    if self.algo == "adafactor":
                        if w.dim() >= 2:
                            st["r"] = torch.zeros(w.shape[0], dtype=torch.float32, device=self.dev)
                            st["c"] = torch.zeros(w.shape[1], dtype=torch.float32, device=self.dev)
                        else:
                            st["v"] = torch.zeros_like(w, dtype=torch.float32)
                    else:
                        st["m"] = torch.zeros_like(w, dtype=self.state_dtype)
                        st["v"] = torch.zeros_like(w, dtype=self.state_dtype)
                    if self.rounding == "kahan":
                        st["comp"] = torch.zeros_like(w_host)  # host-resident
                st["t"] += 1
                t = st["t"]
                if self.algo == "adafactor":
                    b2 = 1.0 - t ** -0.8
                    g2 = g * g + 1e-30
                    if w.dim() >= 2:
                        st["r"].mul_(b2).add_(g2.mean(dim=1), alpha=1 - b2)
                        st["c"].mul_(b2).add_(g2.mean(dim=0), alpha=1 - b2)
                        vhat = torch.outer(st["r"], st["c"]) / st["r"].mean()
                    else:
                        st["v"].mul_(b2).add_(g2, alpha=1 - b2)
                        vhat = st["v"]
                    u = g / vhat.sqrt()
                    u.div_(max(1.0, (u.pow(2).mean().sqrt() / 1.0).item()))
                else:
                    b1, b2 = 0.9, 0.999
                    m, v = st["m"], st["v"]
                    m32 = m.float().mul_(b1).add_(g, alpha=1 - b1)
                    v32 = v.float().mul_(b2).addcmul_(g, g, value=1 - b2)
                    m.copy_(m32)
                    v.copy_(v32)
                    u = (m32 / (1 - b1 ** t)) / ((v32 / (1 - b2 ** t)).sqrt() + 1e-8)
                p32 = w.float().sub_(u, alpha=lr)
                comp = st.get("comp")
                comp_d = comp.to(self.dev) if comp is not None else None
                _round_into(p32, w, self.rounding, comp_d)
                if w is not w_host:
                    w_host.copy_(w)
                    if comp is not None:
                        comp.copy_(comp_d)


o = args.optim
if o == "torch_adamw":
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
elif o == "torch_adafactor":
    opt = torch.optim.Adafactor(params, lr=args.lr)
elif o == "sgd":
    opt = torch.optim.SGD(params, lr=args.lr)
elif o.startswith("af_"):
    opt = LowMemOpt(params, args.lr, "adafactor", o[3:], device=args.opt_device)
elif o == "adamw_sr":
    opt = LowMemOpt(params, args.lr, "adamw", "sr", state_dtype=torch.bfloat16, device=args.opt_device)
elif o == "adamw_nearest_bf16state":
    opt = LowMemOpt(params, args.lr, "adamw", "nearest", state_dtype=torch.bfloat16)
else:
    raise SystemExit(f"unknown optim {o}")

# fixed overfit batches
texts = [tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                 tokenize=False) for q, a in ROWS]
enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.seq)
labels = enc["input_ids"].clone()
labels[enc["attention_mask"] == 0] = -100
nb = len(ROWS) // args.batch

snap = None
if args.precision_check:
    snap = [local(p).detach().clone() for p in params]

torch.cuda.reset_peak_memory_stats()
mon.reset_phase()
losses, times, grad_bytes, phases = [], [], None, []
model.train()
for step in range(args.steps):
    i = step % nb
    sl = slice(i * args.batch, (i + 1) * args.batch)
    t0 = time.perf_counter()
    out = model(input_ids=enc["input_ids"][sl].cuda(), attention_mask=enc["attention_mask"][sl].cuda(),
                labels=labels[sl].cuda())
    torch.cuda.synchronize(); t1 = time.perf_counter(); r1 = psutil.Process().memory_info().rss
    out.loss.backward()
    torch.cuda.synchronize(); t2 = time.perf_counter(); r2 = psutil.Process().memory_info().rss
    if grad_bytes is None:
        gl = [local(p.grad) for p in params if p.grad is not None]
        grad_bytes = sum(g.numel() * g.element_size() for g in gl)
        receipt["grad_device"] = str(gl[0].device) if gl else None
        receipt["grad_dtype"] = str(gl[0].dtype) if gl else None
    opt.step()
    t3 = time.perf_counter(); r3 = psutil.Process().memory_info().rss
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
    phases.append({"fwd_s": round(t1 - t0, 2), "bwd_s": round(t2 - t1, 2), "opt_s": round(t3 - t2, 2),
                   "rss_fwd": round(r1 / GB, 2), "rss_bwd": round(r2 / GB, 2), "rss_opt": round(r3 / GB, 2)})
    losses.append(round(out.loss.item(), 4))
    if step == 0 and snap is not None:
        changed = sum(int((local(p) != s).sum()) for p, s in zip(params, snap))
        receipt["pct_params_changed_step1"] = round(100.0 * changed / n, 3)
        # same check on the mean |update| relative to |w| for context
        del snap
        snap = None

state_bytes = 0
for st in opt.state.values():
    for v in st.values():
        if isinstance(v, torch.Tensor):
            v = local(v)
            state_bytes += v.numel() * v.element_size()
param_bytes = sum(local(p).numel() * local(p).element_size() for p in params)
receipt.update(
    losses=losses,
    phases=phases[:3],
    s_per_step=round(sum(times[1:]) / max(1, len(times) - 1), 3),
    host_param_B_per_param=round(param_bytes / n, 3),
    host_grad_B_per_param=round((grad_bytes or 0) / n, 3),
    host_optstate_B_per_param=round(state_bytes / n, 3),
    peak_vram_alloc_gb=round(torch.cuda.max_memory_allocated() / GB, 3),
    peak_vram_reserved_gb=round(torch.cuda.max_memory_reserved() / GB, 3),
    peak_rss_train_gb=round(mon.phase_peak / GB, 3),
    peak_rss_gb=round(mon.peak / GB, 3),
)
print("PROBE_RECEIPT " + json.dumps(receipt), flush=True)
dist.destroy_process_group()
