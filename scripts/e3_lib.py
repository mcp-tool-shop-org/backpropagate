"""Shared helpers for the E3 VRAM sweep (``pod_e3_sweep.py``) and refit (``e3_refit.py``).

Nothing here needs a GPU, and nothing at import time needs torch: the planner,
the cost model, the budget guard and the receipt scrubber are plain Python so
the tests (and the lead, before paying for a pod) can run them anywhere.

Contents
--------
* the sweep grid: presets (small to large), batches, Unsloth arms, the priority
  tier of every point and the budget guard's drop order;
* a cost model fitted to the stage d timings (see ``estimate_point_seconds``);
* ``fetch_arch``: a model's architecture and *text-only* parameter count from its
  config alone, on the meta device (no weights are ever downloaded);
* ``scrub``: removes a secret (the HF token) from any string headed for a receipt
  or a log.

Standards compliance (workflow-standards.md)
  PIN_PER_STEP 2: every point records git SHA, image, library versions, seed,
  preset and every Trainer argument; the grid is a pure function of its inputs.
  ANDON_AUTHORITY 2: the orchestrator halts on a dead GPU or a missing Unsloth
  extra before spending, and the budget guard stops the run at the cap.
  NAMED_COMPENSATORS 2: the only irreversible acts are pod creation (the lead:
  ``pod.sh delete`` / dead-man timer) and HF-cache deletes, which are
  re-downloadable; no publish, no push.
  DECOMPOSE_BY_SECRETS 2: planning / measuring / fitting are three files that
  share only the receipt schema.
  UNCERTAINTY_GATED_HUMANS 2: the refit proposes, the lead applies; the gate is
  pre-registered in the receipt README.
  EXTERNAL_VERIFIER n/a: no specialized claims (the gate is a numerical rule on
  measured data).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Iterable, Sequence
from typing import Any

GIB = 2**30

#: Batch sizes swept at every preset. 6 is what ``_detect_batch_size`` picks on
#: a 32 GB card, 1 is the floor ``oom_recovery`` would halve down to.
BATCHES: tuple[int, ...] = (1, 2, 4, 6)

#: Batches the budget guard protects first (the latent auto-batch question: the
#: floor, and what the library resolves on a 32 GB card).
PROTECTED_BATCHES: tuple[int, ...] = (1, 6)

#: Steps per point. Step 1 builds the (lazily allocated) paged optimizer state,
#: so the activation + optimizer peak first coexists in step 2 and the allocator
#: settles by step 3-4. Eight steps leave at least four steady-state steps for
#: the median s/step and a plateau check on the NVML peak (see ``plateau_ok``).
DEFAULT_STEPS = 8

#: Planning figure only (pod price per hour for an RTX 5090; the handoff's number).
DEFAULT_USD_PER_HOUR = 0.90

#: The sweep's share of the pod C cap ($2.00 shared with E1 validation and the
#: Llama-3.1-8B smoke, handoff section 4a). The remaining $0.80 is E1's.
DEFAULT_BUDGET_USD = 1.20

#: Planning figure only: download throughput on a RunPod host.
DEFAULT_DL_MBPS = 200.0


# ----------------------------------------------------------------- the grid
@dataclasses.dataclass(frozen=True)
class Preset:
    name: str
    model_id: str
    lora_r: int
    window: int
    packing: bool
    #: parameter count (billions) the library derives from the preset name
    params_b_name: float


def load_presets() -> list[Preset]:
    """Every ``MODEL_PRESETS`` entry, ordered small to large.

    The order is the order they are measured in: weights are deleted from the
    HF cache after each preset, so the cache never holds more than the current
    preset (plus the next one, prefetched in the background).
    """
    from backpropagate.config import MODEL_PRESETS
    from backpropagate.trainer import _estimate_param_count_billions

    out = []
    for name, p in MODEL_PRESETS.items():
        pb = _estimate_param_count_billions(name)
        if pb is None:
            raise RuntimeError(f"cannot derive a parameter count for preset {name!r}")
        out.append(Preset(
            name=name, model_id=p.model_id, lora_r=p.recommended_lora_r,
            window=p.recommended_max_seq_length, packing=p.recommended_packing,
            params_b_name=pb,
        ))
    out.sort(key=lambda p: (p.params_b_name, p.name))
    return out


def size_class(params_b_name: float) -> str:
    """large (14B+), mid (7-13B) or small (under 7B)."""
    if params_b_name >= 14:
        return "large"
    if params_b_name >= 7:
        return "mid"
    return "small"


#: Priority tiers; a lower number is protected longer. The 14B/24B/32B presets
#: at batch 1 and 6 (the latent auto-batch question) come first. Then every
#: other preset at batch 1 and 6 (breadth: the fit needs every architecture and
#: both ends of the batch range before it needs interpolation points), then the
#: batch 2 and 4 points, large first.
_TIER = {
    ("large", True): 0, ("mid", True): 1, ("small", True): 2,
    ("large", False): 3, ("mid", False): 4, ("small", False): 5,
}


def tier_of(params_b_name: float, batch: int) -> int:
    return _TIER[(size_class(params_b_name), batch in PROTECTED_BATCHES)]


@dataclasses.dataclass(frozen=True)
class Point:
    preset: str
    model_id: str
    batch: int
    unsloth: bool
    tier: int
    #: position in execution order (preset small to large, batch ascending, off then on)
    order: int

    @property
    def tag(self) -> str:
        return point_tag(self.preset, self.batch, self.unsloth)


def point_tag(preset: str, batch: int, unsloth: bool) -> str:
    return f"e3_{preset}_b{batch}_{'unsloth' if unsloth else 'plain'}"


def build_points(
    presets: Sequence[Preset],
    batches: Sequence[int] = BATCHES,
    unsloth_arms: Sequence[bool] = (False, True),
) -> list[Point]:
    """The full grid in execution order."""
    pts: list[Point] = []
    for p in presets:
        for b in batches:
            for u in unsloth_arms:
                pts.append(Point(p.name, p.model_id, b, u, tier_of(p.params_b_name, b), len(pts)))
    return pts


def drop_order(points: Iterable[Point]) -> list[Point]:
    """The order the budget guard drops points in (first element dropped first).

    Highest tier number first (small presets at batch 2/4), then, within a tier,
    Unsloth-on points before Unsloth-off (the default Trainer on a box without
    the extra is Unsloth-off), then the later execution position first. The
    14B/24B/32B points at batch 1 and 6 (tier 0) are dropped last of all.
    """
    return sorted(points, key=lambda p: (-p.tier, 0 if p.unsloth else 1, -p.order))


# --------------------------------------------------------------- cost model
#: Approximate bf16-repo sizes in billions of parameters, for the DOWNLOAD
#: estimate only (Hub ``safetensors.total``; the exact figures are recorded by
#: ``fetch_arch`` on the day). Qwen3.5-4B includes its vision tower and MTP head.
REPO_PARAMS_B: dict[str, float] = {
    "llama-3.2-1b": 1.24, "qwen2.5-3b": 3.09, "llama-3.2-3b": 3.21, "smollm3-3b": 3.08,
    "phi-4-mini-3.8b": 3.84, "qwen3.5-4b": 4.66, "qwen2.5-7b": 7.62, "mistral-7b": 7.25,
    "llama-3.1-8b": 8.03, "qwen2.5-14b": 14.77, "mistral-small-24b": 23.57, "qwen2.5-32b": 32.76,
}

# Stage d (RTX 5090, QLoRA, batch 4, 512 tokens): 3B 0.41 s/step, 7B 0.51 s/step,
# i.e. ~5,000 and ~4,000 tokens/s, and the 7B figure is within 25% of the card's
# dense bf16 peak for 6N FLOPs/token (fwd + activation-grad + recompute). Beyond
# that the card is compute bound: tokens/s ~ 30,400 / params_b.
_TOK_S_CAP = 5000.0
_TOK_S_NUM = 30400.0
_STEP_FIXED_S = 0.15
_PROC_START_S = 25.0      # python + torch + transformers + trl + bitsandbytes imports
_LOAD_FIXED_S = 12.0      # stage d: 14.8 s (3B) and 16.2 s (7B) from a warm cache
_LOAD_PER_B_S = 1.5       # nf4 quantize-on-load, extrapolated beyond 7B (no 14B+ timing yet)
_UNSLOTH_LOAD_S = 25.0    # Unsloth patching / kernel warm-up on top
_TEARDOWN_S = 6.0


def repo_params_b(preset: str, fallback: float) -> float:
    return REPO_PARAMS_B.get(preset, fallback * 1.05)


def estimate_point_seconds(preset: Preset, batch: int, unsloth: bool, steps: int = DEFAULT_STEPS) -> float:
    """Wall-clock estimate for one point, in seconds (planning figure, not a promise)."""
    b = repo_params_b(preset.name, preset.params_b_name)
    tok_s = min(_TOK_S_CAP, _TOK_S_NUM / b)
    step_s = batch * preset.window / tok_s + _STEP_FIXED_S
    if unsloth:
        step_s *= 0.8  # Unsloth is faster per token; the planning figure keeps a margin
    load_s = _LOAD_FIXED_S + _LOAD_PER_B_S * b + (_UNSLOTH_LOAD_S if unsloth else 0.0)
    return _PROC_START_S + load_s + steps * step_s + _TEARDOWN_S


def estimate_download_seconds(preset: Preset, mbps: float = DEFAULT_DL_MBPS) -> float:
    gb = repo_params_b(preset.name, preset.params_b_name) * 2.0  # bf16
    return gb * 1000.0 / mbps


@dataclasses.dataclass
class Plan:
    keep: list[Point]
    dropped: list[Point]
    est_seconds: float
    est_usd: float
    budget_seconds: float
    per_preset_seconds: dict[str, float]
    download_seconds: float
    notes: list[str]


def estimate_total_seconds(
    presets: Sequence[Preset],
    points: Sequence[Point],
    steps: int = DEFAULT_STEPS,
    dl_mbps: float = DEFAULT_DL_MBPS,
    prefetch: bool = True,
) -> tuple[float, dict[str, float], float]:
    """(total seconds, measuring seconds per preset, exposed download seconds).

    With ``prefetch`` the next preset downloads while the current one measures,
    so only the first download and any excess over the previous preset's
    measuring time is paid for in wall-clock.
    """
    by = {p.name: p for p in presets}
    per: dict[str, float] = {}
    for pt in points:
        per[pt.preset] = per.get(pt.preset, 0.0) + estimate_point_seconds(by[pt.preset], pt.batch, pt.unsloth, steps)
    order = [p.name for p in presets if p.name in per]
    exposed = 0.0
    prev_measure = 0.0
    for i, name in enumerate(order):
        dl = estimate_download_seconds(by[name], dl_mbps)
        if not prefetch or i == 0:
            exposed += dl
        else:
            exposed += max(0.0, dl - prev_measure)
        prev_measure = per[name]
    return sum(per.values()) + exposed, per, exposed


def plan_budget(
    presets: Sequence[Preset],
    points: Sequence[Point],
    budget_usd: float = DEFAULT_BUDGET_USD,
    usd_per_hour: float = DEFAULT_USD_PER_HOUR,
    steps: int = DEFAULT_STEPS,
    dl_mbps: float = DEFAULT_DL_MBPS,
    prefetch: bool = True,
    overhead_seconds: float = 600.0,
) -> Plan:
    """Fit the grid into the budget by dropping points in ``drop_order``.

    ``overhead_seconds`` covers pod boot, the pip install and preflight, which
    the sweep pays before its first point.
    """
    budget_s = budget_usd / usd_per_hour * 3600.0
    keep = list(points)
    dropped: list[Point] = []
    notes: list[str] = []
    victims = drop_order(points)
    total, per, exposed = estimate_total_seconds(presets, keep, steps, dl_mbps, prefetch)
    while total + overhead_seconds > budget_s and victims:
        v = victims.pop(0)
        keep.remove(v)
        dropped.append(v)
        total, per, exposed = estimate_total_seconds(presets, keep, steps, dl_mbps, prefetch)
    if total + overhead_seconds > budget_s:
        notes.append("even an empty grid exceeds the budget; nothing can run")
    total += overhead_seconds
    return Plan(keep=keep, dropped=dropped, est_seconds=total, est_usd=total / 3600.0 * usd_per_hour,
                budget_seconds=budget_s, per_preset_seconds=per, download_seconds=exposed, notes=notes)


def runtime_guard(
    rest: Sequence[Point],
    by_preset: dict[str, Preset],
    steps: int,
    elapsed_s: float,
    budget_s: float,
    scale: float = 1.0,
) -> list[Point]:
    """Points to drop NOW so that ``elapsed + remaining estimate`` fits the budget.

    ``rest`` is every point not yet run, including the one about to start.
    ``scale`` is observed/estimated seconds so far (so a slow pod shrinks the
    plan). Victims come in ``drop_order``; the tier-0 points go last.
    """
    live = list(rest)

    def need() -> float:
        return scale * sum(estimate_point_seconds(by_preset[r.preset], r.batch, r.unsloth, steps) for r in live)

    dropped: list[Point] = []
    for v in drop_order(rest):
        if elapsed_s + need() <= budget_s:
            break
        live.remove(v)
        dropped.append(v)
    return dropped


# ------------------------------------------------------------------ secrets
def scrub(text: str, secrets: Iterable[str | None] = ()) -> str:
    """Replace every non-trivial secret (and the usual token shapes) in ``text`` with ``***``."""
    import re

    out = text
    for s in secrets:
        if s and len(s) >= 8:
            out = out.replace(s, "***")
    return re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "***", out)


def hf_secrets() -> list[str]:
    """The HF token value(s) in this process's environment, for ``scrub`` only."""
    return [v for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
            if (v := os.environ.get(k))]


# ------------------------------------------------------------- architecture
def fetch_arch(model_id: str) -> dict[str, Any]:
    """Architecture + text-only parameter count of ``model_id``, from its config alone.

    The model is instantiated on the meta device (no storage, no weights
    downloaded), through the same class the library's text-only loader resolves
    (``AutoModelForCausalLM``'s mapping, including its swap of a composite
    config for its text config). ``text_params`` is therefore the parameter
    count the library actually loads; ``vision_params`` is whatever a
    ``AutoModelForImageTextToText`` composite would add.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id)
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg

    loaded_class = None
    text_params = None
    try:
        from transformers.models.auto.auto_factory import _get_model_class

        mc = _get_model_class(cfg, AutoModelForCausalLM._model_mapping)
        use = cfg.get_text_config() if mc.config_class == getattr(cfg, "sub_configs", {}).get("text_config") else cfg
        with torch.device("meta"):
            model = mc(use)
        loaded_class = type(model).__name__
        names = [(n, p.numel()) for n, p in model.named_parameters()]
        text_params = sum(c for _, c in names)
        vision_in_text = sum(c for n, c in names if "visual" in n or "vision" in n)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        vision_in_text = None
        loaded_class = f"unresolved: {type(exc).__name__}: {str(exc)[:200]}"

    composite_params = None
    try:
        from transformers import AutoModelForImageTextToText
        from transformers.models.auto.auto_factory import _get_model_class

        if hasattr(cfg, "vision_config") and getattr(cfg, "vision_config", None) is not None:
            mc2 = _get_model_class(cfg, AutoModelForImageTextToText._model_mapping)
            with torch.device("meta"):
                m2 = mc2(cfg)
            composite_params = sum(p.numel() for p in m2.parameters())
    except Exception:  # noqa: BLE001  # nosec B110 - composite count is informational only
        composite_params = None

    layer_types = getattr(text_cfg, "layer_types", None)
    full_attn = linear_attn = None
    if layer_types:
        full_attn = sum(t == "full_attention" for t in layer_types)
        linear_attn = sum(t == "linear_attention" for t in layer_types)

    hidden = int(text_cfg.hidden_size)
    layers = int(text_cfg.num_hidden_layers)
    heads = int(text_cfg.num_attention_heads)
    vocab = int(text_cfg.vocab_size)
    tie = bool(getattr(text_cfg, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", False)))
    return {
        "model_id": model_id,
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "loaded_class": loaded_class,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": getattr(text_cfg, "num_key_value_heads", None),
        "intermediate_size": getattr(text_cfg, "intermediate_size", None),
        "head_dim": getattr(text_cfg, "head_dim", None) or hidden // max(1, heads),
        "vocab_size": vocab,
        "tie_word_embeddings": tie,
        "full_attention_layers": full_attn,
        "linear_attention_layers": linear_attn,
        "text_params": text_params,
        "text_params_b": None if text_params is None else text_params / 1e9,
        "vision_params_in_text_class": vision_in_text,
        "composite_params_incl_vision": composite_params,
    }


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)
