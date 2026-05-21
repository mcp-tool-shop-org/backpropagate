"""
Backpropagate - Model Card Generation
======================================

F-004: emit a ``model_card.md`` alongside every export so the produced
artefact ships with provenance + reproduction instructions. Follows the
Hugging Face model-card schema documented at
https://huggingface.co/docs/hub/model-cards so the card doubles as the
HF repo's ``README.md`` when pushed via :func:`backpropagate.export.push_to_hub`.

Usage:
    from backpropagate.model_card import generate_model_card

    card_md = generate_model_card(
        run_id="abc123",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        dataset_path="data.jsonl",
        final_loss=0.42,
        loss_history=[1.2, 0.9, 0.6, 0.42],
        steps=100,
    )

    Path("model_card.md").write_text(card_md, encoding="utf-8")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "generate_model_card",
    "build_loss_sparkline",
    "infer_model_short_name",
    "write_model_card_for_export",
]


# Unicode sparkline blocks, low → high.
_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def build_loss_sparkline(loss_history: list[float] | None, width: int = 40) -> str:
    """Render a unicode sparkline for the loss curve.

    Returns an empty string when there's nothing useful to display.
    The sparkline is downsampled to at most ``width`` points so the
    visual stays readable for long training runs.
    """
    if not loss_history:
        return ""

    losses = [float(x) for x in loss_history if isinstance(x, (int, float))]
    if not losses:
        return ""

    # Downsample to width.
    if len(losses) > width:
        step = len(losses) / width
        sampled = [losses[int(i * step)] for i in range(width)]
    else:
        sampled = losses

    lo = min(sampled)
    hi = max(sampled)
    span = hi - lo
    if span <= 0:
        # Flat line — render a single low-block per sample.
        return _SPARKLINE_CHARS[0] * len(sampled)

    chars = []
    last_idx = len(_SPARKLINE_CHARS) - 1
    for value in sampled:
        normalized = (value - lo) / span
        idx = int(normalized * last_idx)
        idx = max(0, min(last_idx, idx))
        chars.append(_SPARKLINE_CHARS[idx])
    return "".join(chars)


def infer_model_short_name(base_model: str | None) -> str:
    """Derive a short, human-friendly name from the base model identifier.

    ``unsloth/Qwen2.5-7B-Instruct-bnb-4bit`` → ``Qwen2.5-7B-Instruct-bnb-4bit-finetune``.
    Falls back to ``"backpropagate-finetune"`` when the base model is unknown.
    """
    if not base_model:
        return "backpropagate-finetune"
    last = base_model.rsplit("/", 1)[-1]
    if not last:
        return "backpropagate-finetune"
    return f"{last}-finetune"


def _format_value(value: Any, none_placeholder: str = "*(not recorded)*") -> str:
    """Render a value for the markdown property table, with a placeholder for None."""
    if value is None:
        return none_placeholder
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_duration(seconds: float | int | None) -> str:
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return "*(not recorded)*"
    s = float(seconds)
    if s < 60:
        return f"{s:.1f} seconds"
    if s < 3600:
        return f"{s / 60:.1f} minutes"
    return f"{s / 3600:.2f} hours"


def generate_model_card(
    *,
    run_id: str | None = None,
    base_model: str | None = None,
    dataset_path: str | None = None,
    dataset_hash: str | None = None,
    final_loss: float | None = None,
    loss_history: list[float] | None = None,
    steps: int | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    seed: int | None = None,
    training_duration: float | None = None,
    gpu_used: str | None = None,
    quantization: str | None = None,
    export_format: str | None = None,
    created_at: str | None = None,
    library_version: str | None = None,
    incomplete_provenance: bool = False,
    extra_tags: list[str] | None = None,
) -> str:
    """Generate a Hugging Face-style model card as a Markdown string.

    The frontmatter follows the HF schema (``base_model`` / ``library_name``
    / ``tags``). The body is operator-readable and human-friendly: a
    property table, an ASCII sparkline of the loss curve, the Stage B/C/D
    + Ship Gate trust signals, and a reproduce-this-run command.

    None-valued fields render as ``*(not recorded)*`` so the card never
    crashes on a partial run — when the run is recorded via
    RunHistoryManager every field should be populated.
    """
    if created_at is None:
        created_at = datetime.now().isoformat(timespec="seconds")
    if library_version is None:
        try:
            from importlib.metadata import version as _pkg_version

            library_version = _pkg_version("backpropagate")
        except Exception:
            library_version = "unknown"

    model_short = infer_model_short_name(base_model)

    tags = ["llm", "fine-tuned", "lora"]
    if quantization:
        tags.append("gguf")
    if extra_tags:
        for tag in extra_tags:
            if tag and tag not in tags:
                tags.append(tag)

    # Frontmatter.
    lines: list[str] = ["---"]
    if base_model:
        lines.append(f"base_model: {base_model}")
    lines.append("library_name: backpropagate")
    lines.append("tags:")
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {model_short}")
    lines.append("")
    if base_model:
        lines.append(
            f"Fine-tuned `{base_model}` via "
            f"[backpropagate](https://github.com/mcp-tool-shop-org/backpropagate) "
            f"v{library_version}."
        )
    else:
        lines.append(
            f"Fine-tuned via "
            f"[backpropagate](https://github.com/mcp-tool-shop-org/backpropagate) "
            f"v{library_version}."
        )

    if incomplete_provenance:
        lines.append("")
        lines.append(
            "> :warning: **Incomplete provenance.** This card was emitted "
            "without a matching RunHistoryManager record; some fields below "
            "fall back to defaults."
        )

    lines.append("")
    lines.append("## Training details")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    lines.append(f"| Run ID | `{run_id or '(not recorded)'}` |")
    lines.append(f"| Base model | `{base_model or '(not recorded)'}` |")
    dataset_line = "*(not recorded)*"
    if dataset_path:
        if dataset_hash:
            dataset_line = f"`{dataset_path}` (sha256: `{dataset_hash}`)"
        else:
            dataset_line = f"`{dataset_path}`"
    elif dataset_hash:
        dataset_line = f"(remote) sha256: `{dataset_hash}`"
    lines.append(f"| Dataset | {dataset_line} |")
    lines.append(f"| Steps | {_format_value(steps)} |")
    lines.append(f"| Final loss | {_format_value(final_loss)} |")
    lines.append(f"| LoRA rank | {_format_value(lora_r)} |")
    lines.append(f"| LoRA alpha | {_format_value(lora_alpha)} |")
    lines.append(f"| Seed | {_format_value(seed)} |")
    lines.append(f"| Training duration | {_format_duration(training_duration)} |")
    lines.append(f"| GPU | {_format_value(gpu_used)} |")
    if export_format:
        lines.append(f"| Export format | `{export_format}` |")
    if quantization:
        lines.append(f"| Quantization | `{quantization}` |")
    lines.append(f"| Created | {created_at} |")
    lines.append(f"| Library version | `backpropagate=={library_version}` |")

    # Loss sparkline.
    sparkline = build_loss_sparkline(loss_history)
    if sparkline:
        lines.append("")
        lines.append("## Loss curve")
        lines.append("")
        lines.append(f"```text\n{sparkline}\n```")
        lines.append("")
        lines.append(
            f"({len(loss_history or [])} loss samples; rendered start → end, "
            "low blocks = lower loss.)"
        )
    elif loss_history is not None:
        lines.append("")
        lines.append("## Loss curve")
        lines.append("")
        lines.append("*(no loss samples recorded)*")

    # Trust signals (Ship Gate / Stage B/C/D residuals).
    lines.append("")
    lines.append("## Trust signals (backpropagate Ship Gate)")
    lines.append("")
    lines.append(
        "- Sigstore provenance via OIDC trusted publishing on every release "
        "(npm + GitHub Releases)."
    )
    lines.append(
        "- Stable error codes (INPUT_/CONFIG_/DEP_/RUNTIME_/STATE_/PARTIAL_) — "
        "every BackpropagateError carries `code`/`message`/`hint`/`cause`."
    )
    lines.append(
        "- `run_id` correlation token across logs, on-disk run history, "
        "checkpoint manifests, and SLAO merge_history."
    )
    lines.append(
        "- Atomic checkpoint writes (B-006) + HF Hub transient retry (B-017) + "
        "Unsloth → transformers fallback (B-010)."
    )
    lines.append("- Test suite: 1800+ tests, 50% coverage floor.")

    # Reproduce block.
    if base_model:
        reproduce_lines = [
            "## Reproduce",
            "",
            "```bash",
            "backprop train \\",
            f"  --model {base_model} \\",
            f"  --data {dataset_path or '<your-dataset>'} \\",
            f"  --steps {steps if steps is not None else '<steps>'} \\",
            f"  --lora-r {lora_r if lora_r is not None else 16}",
            "```",
        ]
        lines.append("")
        lines.extend(reproduce_lines)

    lines.append("")
    return "\n".join(lines)


def write_model_card_for_export(
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    base_model: str | None = None,
    dataset_path: str | None = None,
    dataset_hash: str | None = None,
    final_loss: float | None = None,
    loss_history: list[float] | None = None,
    steps: int | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    seed: int | None = None,
    training_duration: float | None = None,
    gpu_used: str | None = None,
    quantization: str | None = None,
    export_format: str | None = None,
    created_at: str | None = None,
    library_version: str | None = None,
    incomplete_provenance: bool = False,
    extra_tags: list[str] | None = None,
    filename: str = "model_card.md",
) -> Path:
    """Write a ``model_card.md`` next to an export.

    Returns the path that was written. Never raises on partial input —
    missing fields render as ``*(not recorded)*``. IO errors (permission /
    disk full) are logged and re-raised so the caller can decide whether
    to treat them as fatal.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    card_path = output_path / filename

    card_md = generate_model_card(
        run_id=run_id,
        base_model=base_model,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        final_loss=final_loss,
        loss_history=loss_history,
        steps=steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        seed=seed,
        training_duration=training_duration,
        gpu_used=gpu_used,
        quantization=quantization,
        export_format=export_format,
        created_at=created_at,
        library_version=library_version,
        incomplete_provenance=incomplete_provenance,
        extra_tags=extra_tags,
    )

    card_path.write_text(card_md, encoding="utf-8")
    logger.info(f"Model card written to {card_path}")
    return card_path


def load_run_history_for_card(
    output_root: str | Path,
    run_id: str | None,
) -> dict[str, Any] | None:
    """Pull the RunHistoryManager record for a run_id, if available.

    Used by :func:`backpropagate.export` to populate the model card with
    the metadata the trainer / multi-run already recorded on disk. Returns
    ``None`` when the run isn't found (callers fall back to "incomplete
    provenance" mode).
    """
    if not run_id:
        return None
    try:
        from .checkpoints import RunHistoryManager

        manager = RunHistoryManager(str(output_root))
        return manager.get_run(run_id)
    except Exception as exc:
        logger.debug(f"load_run_history_for_card failed: {exc}")
        return None
