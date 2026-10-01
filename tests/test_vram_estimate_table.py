"""E3: ``estimate_vram`` is table-driven, and the refactor changed no output.

The pre-E3 implementation is reproduced below *verbatim* as
``_legacy_estimate_vram`` (copied from ``origin/main`` at 35ca73e, the only edits
being the function name and the one relative import turned absolute). The tests
compare the table-driven ``backpropagate.trainer.estimate_vram`` against it with
exact equality (``==`` on the whole ``VRAMEstimate``, floats included) over a
grid of presets x batch x seq x mode x flags. No constant was refit: the shipped
``VRAMCoefficients`` are the identity table, and a refit needs pod numbers
(``scripts/e3_refit.py``), which is a separate PR.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from backpropagate.config import MODEL_PRESETS
from backpropagate.trainer import (
    _VRAM_ADDEND_TABLE,
    DEFAULT_VRAM_COEFFICIENTS,
    VRAMCoefficients,
    VRAMEstimate,
    _estimate_param_count_billions,
    estimate_vram,
    vram_addends,
)

# ---------------------------------------------------------------------------
# The pre-E3 implementation, verbatim.
# ---------------------------------------------------------------------------

def _legacy_estimate_vram(
    model: str,
    *,
    mode: str = "lora",
    lora_r: int = 16,
    lora_alpha: int | None = None,  # noqa: ARG001 — accepted for symmetry, alpha doesn't affect memory
    batch_size: int = 1,
    gradient_accumulation: int = 1,  # noqa: ARG001 — accepted for symmetry, accum doesn't affect peak VRAM
    max_seq_length: int = 2048,
    bytes_per_param: int = 2,  # bf16 / fp16 default; 4 for fp32, 1 for int8, 0.5 for nf4
    quantize_base: bool = True,  # nf4 base + bf16 adapter (the trainer default)
    hidden_dim: int = 4096,  # 7B-class default; operator can override
    num_layers: int = 32,  # 7B-class default; operator can override
    num_heads: int = 32,  # 7B-class default
    overhead_fraction: float = 0.15,
    param_count_billions: float | None = None,
    offload: bool = False,
    vocab_size: int = 152064,  # offload VRAM model only; Qwen2.5-class default
) -> VRAMEstimate:
    """v1.4 BACKEND-F-002: pre-flight VRAM estimator.

    Returns a structured estimate before ``.train()`` so an operator can
    ask "will this config OOM?" instead of finding out at first OOM. The
    math is back-of-envelope (15% overhead margin); accuracy is within
    ~10-20% of empirical peak for well-known training configs.

    Args:
        model: Model identifier — preset name or HF id. Used to estimate
            parameter count via :func:`_estimate_param_count_billions`.
        mode: ``"lora"`` (default) or ``"full"``. Full FT skips the
            ``lora_adapter_gb`` line + uses higher optimizer state.
        lora_r: LoRA rank (default 16). Ignored when mode='full'.
        lora_alpha: Accepted for API symmetry; does not affect memory.
        batch_size: Per-device batch size.
        gradient_accumulation: Accepted; does not affect peak VRAM.
        max_seq_length: Maximum input sequence length.
        bytes_per_param: 2 for bf16/fp16 (default), 4 for fp32, 1 for
            int8, 0.5 for nf4.
        quantize_base: When True (default — matches the trainer's
            ``load_in_4bit=True``), the base model is in nf4 (0.5 bytes
            per param) and the LoRA adapter is in bf16 (2 bytes per
            param). When False, the base model uses ``bytes_per_param``.
        hidden_dim: Model hidden dim (default 4096 — 7B-class).
        num_layers: Model num_layers (default 32 — 7B-class).
        num_heads: Model num_heads (default 32 — 7B-class).
        overhead_fraction: Fragmentation + framework overhead (default 15%).
        param_count_billions: Optional explicit parameter count. When
            None, estimated via :func:`_estimate_param_count_billions`.

    Returns:
        :class:`VRAMEstimate` carrying the headline number + breakdown.
    """
    notes: list[str] = []

    if param_count_billions is None:
        param_count_billions = _estimate_param_count_billions(model)
    if param_count_billions is None:
        # Defensive default: 7B is the v1.3 canonical 16GB target. Surface
        # the assumption in notes so operators see the imputation.
        param_count_billions = 7.0
        notes.append(
            f"param_count_billions not provided and could not be estimated "
            f"from model={model!r}; assumed 7.0B (v1.3 canonical 16GB target)."
        )

    params = param_count_billions * 1e9
    bytes_to_gb = 1.0 / (1024 ** 3)

    # 1. Model weights. nf4 base when quantize_base=True (the trainer
    #    default with load_in_4bit=True); otherwise use bytes_per_param.
    if quantize_base:
        # nf4: 0.5 bytes per param. The LoRA adapter (if mode='lora')
        # still lives in bf16 — that's the lora_adapter_gb line.
        model_weights_gb = (params * 0.5) * bytes_to_gb
        notes.append("base model quantized to nf4 (0.5 bytes/param)")
    else:
        model_weights_gb = (params * bytes_per_param) * bytes_to_gb

    # 2. LoRA adapter. Per-layer cost = rank * (in_dim + out_dim) * 2 (A + B).
    #    Modern PEFT applies LoRA to ~7 modules per layer (q, k, v, o, gate,
    #    up, down for Llama/Qwen-style architectures). Approximate with a
    #    7-module-per-layer constant.
    if mode == "lora":
        lora_modules_per_layer = 7
        lora_adapter_gb = (
            lora_r
            * (hidden_dim + hidden_dim)  # in + out (typically same)
            * num_layers
            * lora_modules_per_layer
            * bytes_per_param  # adapters in bf16/fp16 even when base is nf4
        ) * bytes_to_gb
    else:
        lora_adapter_gb = 0.0

    # 3. Optimizer state. paged_adamw_8bit (the trainer default on consumer
    #    cards) stores 2 momentum buffers per trainable param at 1 byte each.
    #    Full FT trains the whole model; LoRA only trains the adapter (rank
    #    * (in + out) * num_layers * 7 modules).
    trainable_params: float
    if mode == "lora":
        trainable_params = lora_r * (hidden_dim + hidden_dim) * num_layers * 7
    else:
        trainable_params = params
    # paged 8-bit Adam: 2 buffers * 1 byte + gradient (bytes_per_param)
    optimizer_state_gb = (
        trainable_params * (2 * 1 + bytes_per_param)
    ) * bytes_to_gb

    # 4. Activations. With gradient checkpointing the activation memory
    #    scales as sqrt(num_layers) instead of linearly. Mode='full'
    #    enables gradient_checkpointing=True by default; mode='lora'
    #    inherits the setting from settings.lora.use_gradient_checkpointing.
    activation_layer_factor = (
        max(1.0, num_layers ** 0.5) if mode == "full"
        else float(num_layers)
    )
    activations_gb = (
        batch_size
        * max_seq_length
        * hidden_dim
        * activation_layer_factor
        * bytes_per_param
        * 2  # forward + backward
    ) * bytes_to_gb
    if mode == "full":
        notes.append(
            "mode='full' assumes gradient_checkpointing=True (sqrt(L) "
            "activation memory)"
        )

    # 5. KV cache. batch * seq_len * num_heads * head_dim * num_layers * 2 (k+v)
    #    bytes_per_param-sized. Training rarely keeps the full KV cache (it's
    #    primarily an inference cost) but transformers libraries allocate it
    #    transiently during forward; the constant approximates that share.
    head_dim = hidden_dim // max(1, num_heads)
    kv_cache_gb = (
        batch_size
        * max_seq_length
        * num_heads
        * head_dim
        * num_layers
        * 2  # k + v
        * bytes_per_param
        * 0.25  # Training amortization factor — full cache not retained
    ) * bytes_to_gb

    # 6. v1.7 FSDP2 CPU-offload (mode='full', full_ft_offload=True). Params +
    #    gradients + optimizer state spill into host RAM; the GPU keeps only the
    #    active working set + activations + overhead. Offload full-FT does NOT
    #    quantize the base — host weights are bf16 (2 bytes/param). host_ram_gb
    #    captures the host-resident estimate; the GPU lines shrink accordingly.
    host_ram_gb = 0.0
    if offload and mode == "full":
        # Measured model (backpropagate.offload_engine; the receipts are cited
        # there). Host: bf16 params + grads, ~4.0 B/param, plus the fixed
        # save/reload term. GPU: 2x the root unit (embedding + untied head,
        # assumed untied at vocab_size, which is conservative) + 2 decoder
        # layers + 1.40 MiB per token + 0.8 GiB margin. Nothing optimizer-sized
        # lives on the GPU.
        from backpropagate.offload_engine import (
            _VRAM_MARGIN_GIB,
            _VRAM_MIB_PER_TOKEN,
            offload_host_ram_required_gib,
        )

        host_ram_gb = offload_host_ram_required_gib(params)
        root_bytes = 2 * vocab_size * hidden_dim * 2
        layer_bytes = max(0.0, params - root_bytes / 2) / max(1, num_layers) * 2
        model_weights_gb = (2 * root_bytes + 2 * layer_bytes) * bytes_to_gb
        optimizer_state_gb = 0.0
        activations_gb = batch_size * max_seq_length * _VRAM_MIB_PER_TOKEN / 1024
        kv_cache_gb = 0.0
        overhead_fraction = 0.0
        lora_adapter_gb = 0.0
        model_weights_gb += _VRAM_MARGIN_GIB
        notes.append(
            f"full_ft_offload (measured model): ~{host_ram_gb:.1f} GiB host RAM to train "
            f"+ save; GPU holds the embedding/head + 2 layers + activations "
            f"(PCIe-bound: ~14.7 s/step at 7.6B)"
        )

    subtotal = (
        model_weights_gb
        + lora_adapter_gb
        + optimizer_state_gb
        + activations_gb
        + kv_cache_gb
    )
    overhead_gb = subtotal * overhead_fraction
    total_gb = subtotal + overhead_gb

    return VRAMEstimate(
        total_gb=total_gb,
        model_weights_gb=model_weights_gb,
        lora_adapter_gb=lora_adapter_gb,
        optimizer_state_gb=optimizer_state_gb,
        activations_gb=activations_gb,
        kv_cache_gb=kv_cache_gb,
        overhead_gb=overhead_gb,
        param_count_billions=param_count_billions,
        mode=mode,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        host_ram_gb=host_ram_gb,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Table equality
# ---------------------------------------------------------------------------

_PRESETS = sorted(MODEL_PRESETS)
_ARCH = [  # (hidden, layers, heads): the library default, a 14B-class, a 3B-class
    (4096, 32, 32),
    (5120, 48, 40),
    (2048, 36, 16),
]


def _kw_grid_main():
    """presets x batch x seq x mode x quantize x arch."""
    for preset, batch, seq, mode, quant, (h, nl, nh) in itertools.product(
        _PRESETS, (1, 2, 4, 6, 8), (512, 2048, 4096, 8192), ("lora", "full"),
        (True, False), _ARCH,
    ):
        yield {
            "model": preset, "batch_size": batch, "max_seq_length": seq,
            "mode": mode, "quantize_base": quant,
            "hidden_dim": h, "num_layers": nl, "num_heads": nh,
        }


def _kw_grid_flags():
    """The rarely-moved knobs, over a spread of presets."""
    for preset, bpp, r, ovh, offload, vocab in itertools.product(
        ("qwen2.5-7b", "qwen2.5-32b", "llama-3.2-1b"), (2, 4), (16, 64, 256),
        (0.0, 0.15, 0.3), (False, True), (152064, 128256),
    ):
        for mode in ("lora", "full"):
            yield {
                "model": preset, "bytes_per_param": bpp, "lora_r": r,
                "overhead_fraction": ovh, "offload": offload, "mode": mode,
                "vocab_size": vocab, "batch_size": 3, "max_seq_length": 1024,
            }


def _kw_grid_misc():
    """Unknown model ids (7.0B imputed + a note), explicit param counts."""
    for model, pc in itertools.product(
        ("random/finetune-no-size-clue", "Qwen/Qwen2.5-7B-Instruct-bnb-4bit", ""),
        (None, 0.5, 7.6, 70.0),
    ):
        yield {"model": model, "param_count_billions": pc, "lora_r": 32}


_ALL_CASES = list(_kw_grid_main()) + list(_kw_grid_flags()) + list(_kw_grid_misc())


def test_grid_is_not_trivially_small():
    assert len(_ALL_CASES) >= 3000
    assert len(_PRESETS) == len(MODEL_PRESETS) >= 12


def test_table_driven_estimate_equals_legacy_on_the_whole_grid():
    mismatches = []
    for kw in _ALL_CASES:
        new = estimate_vram(**kw)
        old = _legacy_estimate_vram(**kw)
        if new != old:
            mismatches.append((kw, new, old))
    assert not mismatches, f"{len(mismatches)} of {len(_ALL_CASES)} differ; first: {mismatches[0]}"


def test_use_unsloth_with_default_coefficients_changes_nothing():
    """The Unsloth factors are 1.0 in the shipped table, so the flag is inert."""
    for kw in _ALL_CASES[::7]:
        assert estimate_vram(**kw, use_unsloth=True) == _legacy_estimate_vram(**kw)
        assert estimate_vram(**kw, coefficients=DEFAULT_VRAM_COEFFICIENTS) == _legacy_estimate_vram(**kw)


def test_total_is_exactly_the_sum_of_the_components():
    """total == subtotal + overhead, and the breakdown fields add up the same way."""
    for kw in _ALL_CASES[::11]:
        e = estimate_vram(**kw)
        subtotal = (e.model_weights_gb + e.lora_adapter_gb + e.optimizer_state_gb
                    + e.activations_gb + e.kv_cache_gb)
        assert e.total_gb == subtotal + e.overhead_gb


# ---------------------------------------------------------------------------
# The shipped table is the identity table; nothing was refit
# ---------------------------------------------------------------------------

def test_shipped_coefficients_are_the_identity_table():
    assert DEFAULT_VRAM_COEFFICIENTS.as_dict() == {
        "weights_scale": 1.0, "lora_adapter_scale": 1.0, "optimizer_state_scale": 1.0,
        "activations_scale": 1.0, "kv_cache_scale": 1.0, "embedding_scale": 0.0,
        "logits_scale": 0.0, "fixed_overhead_gb": 0.0, "unsloth_activations_factor": 1.0,
        "unsloth_logits_factor": 1.0,
    }


def test_the_table_names_every_addend_once():
    names = [n for n, _, _ in _VRAM_ADDEND_TABLE]
    assert names == ["weights", "lora_adapter", "optimizer_state", "activations", "kv_cache",
                     "embedding", "logits"]
    fields = {f.name for f in dataclasses.fields(VRAMCoefficients)}
    for _, scale_field, unsloth_field in _VRAM_ADDEND_TABLE:
        assert scale_field in fields
        assert unsloth_field is None or unsloth_field in fields


def test_raw_addends_are_the_unscaled_components():
    """vram_addends() under the identity table IS the breakdown of the estimate."""
    kw = {"model": "qwen2.5-7b", "batch_size": 4, "max_seq_length": 2048, "lora_r": 256}
    e = estimate_vram(**kw)
    raw = vram_addends(
        params=e.param_count_billions * 1e9, lora_r=256, batch_size=4, max_seq_length=2048,
    )
    assert raw["weights"] == e.model_weights_gb
    assert raw["lora_adapter"] == e.lora_adapter_gb
    assert raw["optimizer_state"] == e.optimizer_state_gb
    assert raw["activations"] == e.activations_gb
    assert raw["kv_cache"] == e.kv_cache_gb


# ---------------------------------------------------------------------------
# The table is real: non-default coefficients move the numbers as stated
# ---------------------------------------------------------------------------

def test_a_custom_table_scales_each_addend_and_adds_the_fixed_term():
    kw = {"model": "qwen2.5-14b", "batch_size": 2, "max_seq_length": 4096, "lora_r": 32,
          "hidden_dim": 5120, "num_layers": 48, "num_heads": 40, "vocab_size": 152064}
    base = estimate_vram(**kw)
    raw = vram_addends(
        params=base.param_count_billions * 1e9, lora_r=32, batch_size=2, max_seq_length=4096,
        hidden_dim=5120, num_layers=48, num_heads=40, vocab_size=152064,
    )
    c = VRAMCoefficients(
        weights_scale=1.1, lora_adapter_scale=0.9, optimizer_state_scale=1.2,
        activations_scale=0.5, kv_cache_scale=0.0, embedding_scale=1.5, logits_scale=2.0,
        fixed_overhead_gb=1.25,
    )
    e = estimate_vram(**kw, coefficients=c)
    sub = (raw["weights"] * 1.1 + raw["embedding"] * 1.5 + raw["lora_adapter"] * 0.9
           + raw["optimizer_state"] * 1.2 + raw["activations"] * 0.5 + raw["logits"] * 2.0 + 0.0)
    assert e.total_gb == pytest.approx(sub * 1.15 + 1.25, rel=1e-12)
    assert e.kv_cache_gb == 0.0
    assert any("non-default VRAMCoefficients" in n for n in e.notes)
    assert e != base


def test_unsloth_factors_apply_only_when_unsloth_is_on():
    kw = {"model": "qwen2.5-7b", "batch_size": 4, "max_seq_length": 2048}
    c = VRAMCoefficients(logits_scale=2.0, unsloth_activations_factor=0.5, unsloth_logits_factor=0.0)
    off = estimate_vram(**kw, coefficients=c, use_unsloth=False)
    on = estimate_vram(**kw, coefficients=c, use_unsloth=True)
    assert on.total_gb < off.total_gb
    assert on.activations_gb < off.activations_gb
    assert on.model_weights_gb == off.model_weights_gb


def test_offload_model_ignores_the_coefficients():
    kw = {"model": "qwen2.5-7b", "mode": "full", "offload": True, "batch_size": 1, "max_seq_length": 512}
    c = VRAMCoefficients(weights_scale=3.0, fixed_overhead_gb=9.0, logits_scale=5.0)
    assert estimate_vram(**kw, coefficients=c).total_gb == estimate_vram(**kw).total_gb


def test_estimate_param_count_still_resolves_the_presets():
    for name, preset in MODEL_PRESETS.items():
        assert _estimate_param_count_billions(name) is not None
        assert _estimate_param_count_billions(preset.model_id) is not None


def test_vramestimate_shape_is_unchanged():
    assert [f.name for f in dataclasses.fields(VRAMEstimate)] == [
        "total_gb", "model_weights_gb", "lora_adapter_gb", "optimizer_state_gb",
        "activations_gb", "kv_cache_gb", "overhead_gb", "param_count_billions", "mode",
        "batch_size", "gradient_accumulation", "max_seq_length", "lora_r", "host_ram_gb", "notes",
    ]
