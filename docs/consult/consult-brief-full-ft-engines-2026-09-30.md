# Consult brief: full fine-tuning engines for a single-GPU LLM library

You are being asked for an independent technical review. Below are the
facts and measurements; the questions are at the end. Please challenge
anything that looks wrong, including the measurements themselves.

## 1. Context

`backpropagate` is an open-source Python library (PyTorch, Hugging Face
transformers, TRL, PEFT) for fine-tuning LLMs on **one consumer GPU**. Native
Windows is a first-class platform. The target card is an RTX 5090 (32 GB
VRAM) with 64 GB host RAM; smaller cards (16–24 GB) are supported. The
library's main use is supervised fine-tuning (SFT) followed by export to
GGUF / Ollama. QLoRA is the default and handles up to 32B on a 32 GB card.

The goal under review: **full-parameter fine-tuning of 7B-class models on
one 32 GB card.** Standard full fine-tuning on the GPU (bf16 weights, paged
8-bit AdamW, gradient checkpointing) is capped at about 6B on 32 GB; above
that it does not fit.

## 2. The candidate engines

| | What trains each step | Where the memory goes | Platform |
|---|---|---|---|
| **Standard full FT** (existing) | every weight | all on GPU; ~6 bytes/param + activations | any |
| **Engine A: FSDP2 CPU offload** (built, measured) | every weight | bf16 weights + bf16 grads in host RAM (~4 B/param); optimizer on GPU | Linux / WSL2 (needs NCCL) |
| **Engine B: block-coordinate AdamW** (built, measured on short runs) | one block (a transformer layer, or the embedding/head) for K optimizer steps, then the next | whole model on GPU in bf16; only the active block in fp32 with AdamW state | any, incl. native Windows |
| **Engine C: engine B + host↔GPU block swap** (designed, not built) | as B | frozen blocks stream host→GPU only; active block written back once per visit | any |

Engine A details:
- PyTorch FSDP2 `fully_shard` per decoder layer + root, `CPUOffloadPolicy`,
  `MixedPrecisionPolicy(param_dtype=bf16)`; parameters stay bf16 (the
  earlier route through accelerate upcast every trainable param to fp32,
  ~16 B/param).
- Optimizer: Adafactor, factored second moment, **no momentum**, update
  clipping at RMS 1.0, absolute learning rate. The step runs on the GPU:
  each parameter shard and its gradient are streamed in, updated in fp32,
  written back.
- **No fp32 master weights.** The fp32 result is written back to bf16 with
  stochastic rounding (add uniform random bits below the bf16 mantissa,
  truncate).
- Its own plain SFT loop: no packing, no response-only loss masking, no
  intermediate checkpoints, no resume, gradient accumulation untested above 1.
- Host memory is page-locked in place with `cudaHostRegister`. PyTorch's
  pinned allocator was 3–5× faster but rounds each block up to a power of
  two, which pushed a 7B run past 60 GiB.

Engine B details:
- Modelled on BAdam (Luo, Yu, Li, NeurIPS 2024, arXiv:2404.02827), which
  reports parity with Adam and Llama-3-8B full fine-tuning in 23.5 GB.
- Implemented as a `torch.optim.Optimizer` wrapper inside the stock TRL
  `SFTTrainer`, so packing, checkpoints, resume, gradient accumulation and
  global grad-norm clipping keep working.
- Active block: fp32 master copy under bf16 autocast, fp32 AdamW. On block
  switch the block is written back to bf16 with stochastic rounding and its
  optimizer state is freed. Order: random reshuffle. Frozen blocks keep no
  grads, and backward stops at the lowest block that requires grad.

## 3. Measurements (all on an RTX 5090, 32 GB, single runs unless noted)

### 3.1 Engine A (FSDP2 offload)

| Model | Host RAM peak | VRAM alloc / reserved | s/step (512 tokens) |
|---|---|---|---|
| Qwen2.5-7B-Instruct (7.62B) | 30.8 GiB training; 32.2 GiB incl. save + reload | 5.3 / 14.7 GiB | 14.7 (also 14.7 at 2048 tokens) |
| Qwen3-4B | 25.1 GiB incl. save | — | 4.3 |
| SmolLM3-3B | — | 4.3 GiB at batch 4 | 5.1 |
| Qwen2.5-1.5B | — | — | 1.8 |

- 7B run: 20 steps, loss 0.341 → 0.0067 on a tiny overfit set; save →
  reload → generate works.
- Host RAM fit across 1.5B–7.6B: **3.73 GiB per billion params + 10.1 GiB**
  (fixed term includes save + reload).
- Under emulated VRAM caps: 3B trained under 6 GiB; 4B and 7.6B under 8 GiB.
- Update retention, i.e. how much of the intended fp32 update survives the
  write to bf16 (sum(actual Δ·sign(intended Δ)) / sum(|intended Δ|)),
  measured at 1.5B, step 1:
  - round-to-nearest: 0.171 (the run stopped learning);
  - stochastic rounding: 1.0001.
  - The fraction of parameters that changed did *not* separate the two
    regimes: 8.4% vs 14.8% at 1.5B, and 3.0% for round-to-nearest at 360M.
- Precision A/B at 360M, lr 2e-5, 30 steps (mean of the last 5 losses on an
  overfit set):

  | Setup | Loss |
  |---|---|
  | fp32 AdamW reference | 0.170 |
  | fp32 Adafactor | 0.136 |
  | bf16 Adafactor, round-to-nearest | 0.964 |
  | bf16 Adafactor, stochastic rounding | 0.071 |
  | bf16 Adafactor, Kahan compensation (+2 B/param) | 0.135 |
  | bf16 AdamW | 1.70 |

- Quality on real data. SmolLM3-3B, Dolly-15k (400 train / 150 disjoint
  held-out), 150 steps, batch 4, 512 tokens, lr 2e-5, 1 seed. Held-out
  loss, untrained 2.4546:

  | Arm | Held-out after | s/step |
  |---|---|---|
  | Engine A | 1.932 | 5.1 |
  | Standard full FT | 1.838 | 0.63 |

### 3.2 Engine B (block-coordinate AdamW)

Same Dolly setup, 150 steps, 3 seeds, mean ± sd. The standard full FT
baseline uses paged 8-bit AdamW.

| Model (untrained held-out) | Arm | Held-out after | s/step | Peak alloc / reserved |
|---|---|---|---|---|
| Qwen2.5-1.5B (2.420) | standard full FT | 1.877 ± 0.001 | 0.149 | 6.7 / 7.3 GiB |
| 1.5B | engine B, K=50 | 1.994 ± 0.002 | 0.071 | 8.0 / 10.1 GiB |
| 1.5B | engine B, K=5 | 1.912 ± 0.004 | 0.061 | 8.0 / 10.4 GiB |
| SmolLM3-3B (2.455) | standard full FT | 1.837 ± 0.001 | 0.256 | 12.6 / 13.4 GiB |
| 3B | engine B, K=50 | 2.047 ± 0.043 | 0.096 | 7.3 / 7.8 GiB |
| 3B | engine B, K=5 | 2.015 ± 0.011 | 0.100 | 11.6 / 14.2 GiB |
| Qwen2.5-7B (2.761) | engine B, K=5, random order, 1 seed | 1.720 | 0.17 | 25.4 / 30.4 GiB |
| 7B | engine B, K=30, descending, 1 seed | 2.012 | 0.135 | 25.4 / 28.4 GiB |
| 7B | QLoRA, 1 seed | 1.708 | 0.47 | 12.1 / 12.5 GiB |

- 7B under caps (batch 1): 24 GiB with embeddings trained runs out of
  memory when the embedding/head block activates (the vocabulary matrix is
  0.545B params against 0.233B per layer). 24 GiB with embeddings frozen
  fits (18.0 / 18.6 GiB). 16 GiB runs out of memory.
- Write-back A/B (1 seed, held-out loss after, nearest / stochastic):

  | K | 360M | 3B |
  |---|---|---|
  | 10 | 2.278 / 2.243 | 2.043 / 2.035 |
  | 50 | 2.253 / 2.239 | 2.047 / 2.047 |
  | 200 (single block visit in 150 steps) | 2.262 / 2.260 | 2.024 / 2.024 |

- One 3B layer trained alone for 150 steps (K=200) reached held-out 2.024,
  better than four blocks at K=50 (2.047).

### 3.3 QLoRA presets (for context)

Peak VRAM at each preset's full context window, batch 1, transformers path
(no Unsloth), rank 32, 8-bit AdamW: Qwen2.5-14B at 4096 tokens 25.0 GiB;
Mistral-Small-24B at 4096 tokens 26.5 GiB; Qwen2.5-32B at 2048 tokens
28.8 GiB. The library's analytic VRAM estimator predicted 14.0 / 17.7 /
22.1 GB for these, and 20.1 GB for SmolLM3-3B full fine-tuning at batch 4
(measured 12.6 GiB).

## 4. Literature already consulted (for reference, not as conclusions)

- BAdam, Luo et al. 2024, arXiv:2404.02827: block coordinate descent with
  Adam; memory 2M + 16M/D GB; reported parity with Adam on SFT; its Adam
  baseline overfit at lr 1e-5; SFT only.
- LOMO (arXiv:2306.09782) and AdaLomo (arXiv:2310.10195): fused updates;
  SGD-style lags AdamW, and a second-moment term closes most of the gap.
- Zhang et al. 2024, arXiv:2402.16788: why transformers need adaptive
  optimizers.
- Zamirai et al. 2021, arXiv:2010.06192; Ozkara et al. 2025,
  arXiv:2502.20566: stochastic rounding for bf16 master weights.
- Yu et al. 2024 (Collage), arXiv:2405.03637: Kahan-compensated bf16 did not
  match fp32 in their LLM runs.
- Biderman et al. 2024, arXiv:2405.09673: in standard low-rank settings LoRA
  substantially underperforms full fine-tuning on code and math, and
  forgets less.
- Thinking Machines 2025, "LoRA Without Regret": LoRA matches full
  fine-tuning when applied to all layers and the data fits its capacity.
- Shazeer & Stern 2018, arXiv:1804.04235: Adafactor.
- Dettmers et al. 2021, arXiv:2110.02861: 8-bit optimizers.

## 5. The planned next experiment (critique it)

- Data: GSM8K (train split; a fixed 250-question sample of the test split).
- Metrics: held-out loss on test answers; accuracy of the final number after
  `####` under greedy generation.
- 3B arms: standard full FT, engine B K=5, engine B K=50, QLoRA (rank 256,
  all linear layers). 1000 optimizer steps each, batch 4, 512 tokens, 2
  seeds. Each arm uses its own default learning rate: full FT 2e-5, LoRA
  2e-4.
- 7B arms: engine B K=5 vs QLoRA, 1000 steps, 1 seed.
- Budget: about 1.5 hours on one RTX 5090.

## 6. Questions

1. **Engine B's value.** Given sections 3.2 and 4, is engine B worth
   continuing? At 7B on a 24–32 GB card it is the only fast full-parameter
   option; below 7B it lost to standard full FT. What result from the next
   experiment should make us drop it, and what should make us ship it?
2. **Why the gap to BAdam's reported parity?** Which explanations fit the
   data best? Candidates include the short run (150 steps visits few
   blocks), K, the learning rate, the embedding/head block, the stronger
   baseline, and fp32-master-per-visit plus stochastic write-back. What
   single experiment would discriminate between them most cheaply?
3. **Experiment design (section 5).** Is it fair and sufficient to decide
   question 1? Look specifically at: different learning rates per arm;
   paged 8-bit AdamW in the baseline vs fp32 AdamW in engine B; whether 1000
   steps and 2 seeds are enough; whether GSM8K is the right task to show a
   full-FT-over-LoRA advantage at 3B/7B; whether greedy exact-match on 250
   questions has enough statistical power. What would you change within
   about the same budget?
4. **Engine A's numerics.** With bf16 weights, stochastic rounding and
   momentum-free Adafactor, what failure modes should we expect on long
   runs (thousands of steps, decaying learning rate, weight decay) that
   these short runs cannot show? Is the update-retention measure in 3.1 a
   sound runtime check? What would you monitor instead or as well?
5. **Engine A's speed.** 14.7 s/step at 7.6B does not change between 512 and
   2048 tokens, so it is transfer-bound. Moving ~15 GB of bf16 weights
   host→GPU in forward and again in backward, plus gradients back, at a
   measured ~24 GB/s pinned PCIe 4.0 rate suggests a floor of roughly
   1.3–1.9 s/step. Where is the other ~12 s most likely going, given
   per-layer FSDP2 sharding, CPU offload with `cudaHostRegister`'d
   parameters, pageable gradients, and a GPU-side optimizer that streams
   each parameter and gradient in again?
6. **Anything else.** Is any number in section 3 internally inconsistent or
   implausible? Is anything important missing from the comparison?

## 7. Answer format

For each question: a direct verdict first, then reasoning, then concrete
changes, then your confidence (high / medium / low). When you rely on
published work, name it with an identifier (arXiv id, DOI or URL); say
plainly when something is your own inference.
