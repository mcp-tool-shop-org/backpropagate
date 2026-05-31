"""
Backpropagate - MLX / Apple-Silicon Training Backend (v1.5 T3.1)
================================================================

A second training rail that targets **Apple Silicon** (M-series Macs) via
Apple's ``mlx-lm`` toolchain (``mlx_lm.lora`` for LoRA training,
``mlx_lm.fuse`` for adapter merge + optional GGUF export). It retires the
"no macOS training" boundary (V1_5_BRIEF finding 21): the historical blocker
was CUDA-coupling, not macOS, and unified memory removes the VRAM wall.

⚠️ **BUILT-BUT-UNVERIFIED on Apple Silicon.** ⚠️
``mlx-lm`` is **Apple-Silicon-ONLY** (macOS + arm64) and CANNOT be installed or
exercised on the Windows / CUDA rig this module was authored on. Every
``mlx_lm`` invocation is therefore kept **behind a subprocess seam** — this
module NEVER does ``import mlx_lm`` (it must import cleanly on a host where mlx
is absent), drives the documented ``mlx_lm.lora`` / ``mlx_lm.fuse`` CLIs, and is
covered by **mocked** unit tests. A real-hardware smoke (owned by a sibling
agent) SKIPS on non-Apple hosts. This is the honest dual of the FP8 path's
"experimental, skips on unsupported hardware" discipline: the code is wired and
unit-verified, but the end-to-end MLX run has not been observed on real silicon
as of v1.5. Report anomalies (loss parsing, argv shape, GGUF export) so the rail
can graduate to "verified."

Architecture
------------
* :func:`detect_apple_silicon` — pure, mockable host probe (Darwin + arm64 +
  the ``mlx`` feature flag).
* :func:`resolve_backend` — "auto" → "mlx" on Apple Silicon else "cuda";
  "cuda"/"mlx" returned as-is (the validity of a *forced* "mlx" on a non-Apple
  host is enforced at the Trainer call site, not here).
* :func:`prepare_mlx_data_dir` — converts any supported dataset into the
  ``mlx_lm.lora`` data-directory layout (``train.jsonl`` + optional
  ``valid.jsonl``, ONE chat record per line). Pure file IO.
* :class:`MLXBackend` — builds the ``lora_config.yaml`` dict (the load-bearing
  PEFT-alpha → mlx-absolute-``scale`` mapping lives in :meth:`build_config`),
  the argv, and runs ``mlx_lm.lora`` through the shared subprocess seam
  (:func:`backpropagate.export._run_subprocess_interruptible`). The merged /
  GGUF export route (:meth:`fuse`) is a thin wrapper over ``mlx_lm.fuse``.

The MLX adapter directory is plain safetensors and feeds the existing
``export_ollama_adapter`` path unchanged — no new export code is needed for
v1.5 (DOCS notes this; this module does not build it).
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .feature_flags import check_feature

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .datasets import DatasetLoader as _DatasetLoader  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "detect_apple_silicon",
    "resolve_backend",
    "prepare_mlx_data_dir",
    "MLXRunResult",
    "MLXBackend",
]

# Default timeout (seconds) for the mlx_lm.lora / mlx_lm.fuse subprocess. MLX
# LoRA runs can be long; this matches the generous ceiling export.py uses for
# its own long-running GGUF subprocess. Overridable per call.
_DEFAULT_MLX_TIMEOUT = 24 * 60 * 60  # 24h

# mlx_lm.lora's default LoRA layer count. mlx applies LoRA to the last
# ``num_layers`` transformer blocks (not every linear like our PEFT
# "all-linear" default — DOCS documents the q+v-on-N-layers vs all-linear
# divergence). 16 is the mlx-lm shipped default.
_MLX_DEFAULT_NUM_LAYERS = 16


# Reuse the ChatML turn shape datasets.py emits so we can round-trip a
# converted ChatML string back into the mlx ``chat`` message-list. Mirrors
# ``datasets._CHATML_TURN_RE`` (kept local so this module has no hard import
# coupling to that private regex).
_CHATML_TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)


# =============================================================================
# HOST / BACKEND RESOLUTION
# =============================================================================

def detect_apple_silicon() -> bool:
    """Return True iff this host is an Apple-Silicon Mac WITH the mlx toolchain.

    Three conjuncts, all required:

    * ``platform.system() == "Darwin"`` — macOS,
    * ``platform.machine() == "arm64"`` — Apple Silicon (NOT an Intel Mac),
    * ``check_feature("mlx")`` — the ``mlx_lm`` package is importable (the
      ``[mlx]`` extra is installed; detected via ``find_spec`` only, never an
      actual import).

    Pure + mockable: tests patch ``platform.system`` / ``platform.machine`` and
    ``backpropagate.mlx_backend.check_feature`` to drive every truth-table row
    without real Apple hardware. ``platform`` is imported INSIDE the function so
    a test patching ``platform.system`` via this module's namespace is seen.
    """
    import platform

    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and check_feature("mlx")
    )


def resolve_backend(requested: str) -> str:
    """Resolve the requested backend string to a concrete rail ("cuda"|"mlx").

    * ``"auto"`` → ``"mlx"`` when :func:`detect_apple_silicon` is True, else
      ``"cuda"`` (so existing CUDA rigs are byte-identical).
    * ``"cuda"`` / ``"mlx"`` → returned as-is.

    This function only RESOLVES; it does NOT validate. The cross-field
    "forced ``backend='mlx'`` on a non-Apple host is unrunnable" check belongs
    at the Trainer call site (which can raise the structured
    ``CONFIG_INVALID_SETTING``), not here — keeping resolve_backend a pure,
    side-effect-free mapping that the truth-table tests can exercise directly.
    Any value outside {"auto","cuda","mlx"} is returned unchanged (the config /
    Trainer validators reject bad values upstream; this stays total).
    """
    if requested == "auto":
        return "mlx" if detect_apple_silicon() else "cuda"
    return requested


# =============================================================================
# DATASET → mlx_lm.lora DATA DIRECTORY
# =============================================================================

def _chatml_to_messages(chatml: str) -> list[dict[str, str]]:
    """Parse a ChatML string into an mlx ``chat`` message-list.

    ``DatasetLoader.to_chatml()`` is the universal converter — it normalizes
    ShareGPT / Alpaca / OpenAI / ChatML / raw-text inputs into ChatML strings
    (``<|im_start|>role\\n...<|im_end|>`` turns). mlx_lm.lora's ``chat`` format
    instead wants ``{"messages": [{"role", "content"}, ...]}``, so we round-trip
    the ChatML text back into the role/content message-list. Bodies are
    ``.strip()``-ed (the converter pads each turn with a trailing newline before
    ``<|im_end|>``). Turns whose role is empty are skipped defensively.
    """
    messages: list[dict[str, str]] = []
    for role, body in _CHATML_TURN_RE.findall(chatml):
        if not role:
            continue
        messages.append({"role": role, "content": body.strip()})
    return messages


def prepare_mlx_data_dir(
    dataset: Any,
    out_dir: str | Path,
    *,
    valid_fraction: float = 0.0,
    seed: int = 42,
    max_samples: int = 0,
    shuffle: bool = True,
    reasoning_trace: bool = False,
    min_trace_tokens: int = 8,
    max_trace_tokens: int = 8192,
) -> Path:
    """Materialize ``dataset`` into the mlx_lm.lora data-directory layout.

    Writes ``<out_dir>/train.jsonl`` (and, when ``valid_fraction > 0``,
    ``<out_dir>/valid.jsonl``) with **exactly one** chat record per line —
    ``{"messages": [{"role", "content"}, ...]}`` — using
    ``json.dumps(rec, ensure_ascii=False)`` per line. It NEVER pretty-prints or
    dumps the whole list as one JSON array (that is the mlx data-format
    contract: one example per line).

    Args:
        dataset: Anything :class:`~backpropagate.datasets.DatasetLoader` accepts
            — a file path (JSONL/JSON/CSV/Parquet/txt/md) or an in-memory list
            of samples. Routed through ``DatasetLoader(dataset).to_chatml()`` so
            every supported source format is normalized identically.
        out_dir: Destination directory (created if missing).
        valid_fraction: Fraction of records (0.0–1.0) to split into
            ``valid.jsonl``. 0.0 (default) writes only ``train.jsonl``. The
            split is taken AFTER the optional shuffle so it is a random holdout.
        seed: RNG seed for the shuffle + split (mirrors the SFT path's
            ``settings.training.seed`` so the two rails are reproducible the
            same way).
        max_samples: Cap on total records kept (0 = all). Applied AFTER shuffle
            (mirrors ``_load_dataset``'s shuffle-then-select ordering).
        shuffle: Whether to shuffle before the cap + split (default True,
            matching ``settings.data.shuffle``).
        reasoning_trace: v1.5 T3.2 / re-audit #10. When True, run CORE's
            :func:`datasets.filter_by_trace_length` over the converted ChatML
            rows BEFORE they become mlx ``messages`` records — dropping rows
            whose summed ``<think>`` span is empty / out of
            ``[min_trace_tokens, max_trace_tokens]`` / unbalanced. This makes
            the trace knobs LIVE on the MLX rail (without it, reasoning_trace on
            MLX was a no-op that left empty / over-long / unbalanced traces in).
            Token counting here uses CORE's APPROX counter
            (``_count_tokens_approx``, ~4 chars/token) — the MLX path loads its
            model inside the ``mlx_lm.lora`` subprocess, so there is NO
            in-process tokenizer to count exactly with. CAVEAT: the approx
            counter **under-counts CJK by ~4-8x** (CJK is ~1+ token/char), so
            CJK traces look shorter than they are and can be wrongly dropped as
            too-short against ``min_trace_tokens``; for CJK-heavy reasoning data
            re-derive the bounds against your tokenizer. (The doubled-``<think>``
            chat-template advisory the CUDA rail runs is tokenizer/template
            dependent and therefore CANNOT run here — it stays CUDA-only.)
        min_trace_tokens: Low cutoff for the trace filter (approx tokens).
        max_trace_tokens: High cutoff for the trace filter (approx tokens).

    Returns:
        The ``out_dir`` :class:`~pathlib.Path` (the directory passed to
        ``mlx_lm.lora --data``).

    Raises:
        DatasetFormatError: when conversion yields ZERO usable records (every
            row produced no parseable chat turn, or — with ``reasoning_trace`` —
            the trace filter dropped every row). Raised BEFORE any file write so
            the operator gets a structured ``INPUT_DATASET_FORMAT_UNSUPPORTED``
            naming the format-mismatch cause, instead of a 0-byte ``train.jsonl``
            and a later opaque ``mlx_lm.lora`` subprocess failure (re-audit LOW).

    Pure file IO — fully testable without mlx installed.
    """
    from .datasets import DatasetLoader

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Universal conversion → list[{"text": "<chatml>"}].
    chatml_rows = DatasetLoader(dataset).to_chatml()

    # v1.5 T3.2 / re-audit #10: trace-length filtering on the MLX rail. Run it
    # on the ChatML ``{"text": ...}`` rows (CORE's filter operates on exactly
    # that shape) BEFORE the round-trip into mlx ``messages`` records, so empty
    # / over-long / unbalanced <think> traces are dropped here too — not just on
    # the CUDA path. Uses the approx token counter (no in-process tokenizer on
    # this rail; see the docstring CJK caveat).
    if reasoning_trace:
        from .datasets import filter_by_trace_length

        text_rows = [
            {"text": (row.get("text", "") if isinstance(row, dict) else str(row))}
            for row in chatml_rows
        ]
        kept, stats = filter_by_trace_length(
            text_rows,
            min_trace_tokens=min_trace_tokens,
            max_trace_tokens=max_trace_tokens,
            require_think=True,
            token_counter=None,  # approx (~4 chars/token); CJK under-counts
        )
        logger.info("prepare_mlx_data_dir (reasoning_trace): %s", stats.summary())
        if not kept:
            # Mirror the CUDA rail's loud-on-total-wipeout discipline, but on the
            # MLX rail there is no model loaded yet, so an empty set MUST fail
            # loud BEFORE writing (a 0-byte train.jsonl would surface as an
            # opaque mlx_lm.lora error). Hand off to the shared 0-record raise
            # below by re-pointing chatml_rows at the (empty) kept set.
            logger.warning(
                "prepare_mlx_data_dir: trace-length filtering removed every "
                "row (min=%s, max=%s think tokens, require_think=True). Relax "
                "the trace bounds or confirm the targets carry <think> spans.",
                min_trace_tokens,
                max_trace_tokens,
            )
        chatml_rows = kept

    records: list[dict[str, list[dict[str, str]]]] = []
    for row in chatml_rows:
        text = row.get("text", "") if isinstance(row, dict) else str(row)
        messages = _chatml_to_messages(text)
        if not messages:
            # A row that produced no parseable turn would write an empty
            # conversation mlx would reject — skip it with a breadcrumb.
            logger.warning(
                "prepare_mlx_data_dir: skipped a row that produced no "
                "chat turns (possible format mismatch)."
            )
            continue
        records.append({"messages": messages})

    # Re-audit LOW: refuse to write a 0-record (0-byte train.jsonl) dataset.
    # Zero records means either every row failed ChatML conversion (a format
    # mismatch) or — with reasoning_trace — the trace filter wiped the set.
    # Fail loud HERE with a structured cause instead of letting mlx_lm.lora
    # choke on an empty data dir with an opaque subprocess error.
    if not records:
        from .exceptions import DatasetFormatError

        cause = (
            "every row was dropped by the reasoning-trace filter "
            f"(min={min_trace_tokens}, max={max_trace_tokens} think tokens, "
            "require_think=True) — relax the trace bounds or confirm the "
            "targets carry <think> spans"
            if reasoning_trace
            else "no row produced a parseable ChatML chat turn — the source "
            "format may not be a recognized chat/instruction dataset"
        )
        raise DatasetFormatError(
            "prepare_mlx_data_dir produced 0 usable records, so there is "
            f"nothing to train on ({cause}). Refusing to write an empty "
            "train.jsonl that mlx_lm.lora would later fail on opaquely.",
            detected_format=None,
            supported_formats=[
                "ShareGPT", "Alpaca", "OpenAI chat", "ChatML", "raw text",
            ],
        )

    # Shuffle (seeded) then cap — same ordering as _load_dataset's
    # shuffle-then-select so a samples cap is a random subset, not a head slice.
    if shuffle:
        random.Random(seed).shuffle(records)
    if max_samples and max_samples > 0 and len(records) > max_samples:
        records = records[:max_samples]

    # Optional train/valid split (holdout taken from the tail post-shuffle).
    valid_records: list[dict[str, list[dict[str, str]]]] = []
    train_records = records
    if valid_fraction and valid_fraction > 0.0:
        n_valid = int(len(records) * valid_fraction)
        if n_valid > 0:
            valid_records = records[-n_valid:]
            train_records = records[:-n_valid]

    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        # ONE json object per line — the mlx data-format contract.
        with path.open("w", encoding="utf-8") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False))
                fh.write("\n")

    _write_jsonl(out_path / "train.jsonl", train_records)
    if valid_records:
        _write_jsonl(out_path / "valid.jsonl", valid_records)

    logger.info(
        "prepare_mlx_data_dir: wrote %d train%s record(s) to %s",
        len(train_records),
        f" + {len(valid_records)} valid" if valid_records else "",
        out_path,
    )
    return out_path


# =============================================================================
# RUN RESULT
# =============================================================================

@dataclass
class MLXRunResult:
    """Result of an ``mlx_lm.lora`` run (the subprocess seam's return shape)."""

    adapter_path: str
    final_loss: float | None
    iters: int
    raw_stdout: str = ""
    val_loss: float | None = None


# =============================================================================
# STDOUT PARSING (best-effort)
# =============================================================================

# mlx_lm.lora prints training progress lines like:
#   "Iter 100: Train loss 1.234, Learning Rate 1.000e-05, ..."
#   "Iter 100: Val loss 1.456, Val took 2.3s"
# The exact format varies across mlx-lm versions; these are intentionally loose
# so a minor format drift degrades to final_loss=None (+ WARN) rather than
# crashing the run.
_TRAIN_LOSS_RE = re.compile(r"[Tt]rain loss[:\s]+([0-9]*\.?[0-9]+)")
_VAL_LOSS_RE = re.compile(r"[Vv]al loss[:\s]+([0-9]*\.?[0-9]+)")


def _parse_final_losses(stdout: str) -> tuple[float | None, float | None]:
    """Extract the LAST train-loss and val-loss from mlx_lm.lora stdout.

    Best-effort: returns ``(None, None)`` (caller logs a WARN, does NOT fail)
    when no recognizable loss line is present. Returns the LAST match of each
    so the value reflects the end of training, not the first logged step.
    """
    train_matches = _TRAIN_LOSS_RE.findall(stdout or "")
    val_matches = _VAL_LOSS_RE.findall(stdout or "")
    final_train = float(train_matches[-1]) if train_matches else None
    final_val = float(val_matches[-1]) if val_matches else None
    return final_train, final_val


# =============================================================================
# MLX BACKEND
# =============================================================================

@dataclass
class MLXBackend:
    """Drives ``mlx_lm.lora`` (+ optional ``mlx_lm.fuse``) via a subprocess seam.

    All config assembly (:meth:`build_config`, :meth:`write_config`,
    :meth:`build_argv`) is PURE and unit-tested. Only :meth:`run` / :meth:`fuse`
    touch the (un-runnable on this rig) ``mlx_lm`` CLI, and they do so ONLY
    through :func:`backpropagate.export._run_subprocess_interruptible` — there
    is NO ``import mlx_lm`` anywhere in this module.
    """

    model: str
    dataset_dir: str | Path
    adapter_path: str | Path
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    iters: int
    batch_size: int
    max_seq_length: int
    num_layers: int = _MLX_DEFAULT_NUM_LAYERS
    grad_checkpoint: bool = False
    seed: int = 42
    # Forwarded into _run_subprocess_interruptible; overridable for tests.
    timeout: float = field(default=_DEFAULT_MLX_TIMEOUT)

    def build_config(self) -> dict[str, Any]:
        """Build the ``lora_config.yaml`` dict for ``mlx_lm.lora --config``.

        THE LOAD-BEARING MAPPING (unit-tested):
        ``lora_parameters = {"rank": lora_r,
                             "scale": lora_alpha / lora_r,
                             "dropout": lora_dropout}``

        PEFT expresses adapter strength as ``alpha`` with an effective scale of
        ``alpha / r``; mlx_lm instead takes an ABSOLUTE ``scale`` (its default is
        20.0 — NOT an alpha). So a faithful translation of a PEFT
        ``(r, alpha)`` pair is ``scale = alpha / r`` (e.g. r=256/alpha=512 →
        scale 2.0; r=16/alpha=32 → scale 2.0; r=8/alpha=32 → scale 4.0). This
        preserves the operator's intended adapter magnitude across the two
        frameworks.

        ``lora_parameters.keys`` is intentionally OMITTED so mlx_lm applies its
        own default target (the q+v projections on the last ``num_layers``
        blocks). NB this diverges from the CUDA rail's PEFT "all-linear" default
        target — DOCS documents the q+v-on-N-layers vs all-linear divergence;
        this module does not attempt to force all-linear on mlx (mlx's
        target-module spec differs and is out of scope for v1.5).
        """
        return {
            "model": self.model,
            "train": True,
            "fine_tune_type": "lora",
            "data": str(self.dataset_dir),
            "adapter_path": str(self.adapter_path),
            "iters": int(self.iters),
            "batch_size": int(self.batch_size),
            "max_seq_length": int(self.max_seq_length),
            "learning_rate": float(self.learning_rate),
            "num_layers": int(self.num_layers),
            "grad_checkpoint": bool(self.grad_checkpoint),
            "seed": int(self.seed),
            # The PEFT-alpha → mlx-absolute-scale translation. See docstring.
            "lora_parameters": {
                "rank": int(self.lora_r),
                "scale": self.lora_alpha / self.lora_r,
                "dropout": float(self.lora_dropout),
            },
        }

    def write_config(self, path: str | Path) -> Path:
        """Write :meth:`build_config` to ``path`` as YAML. Returns the Path.

        Uses PyYAML (a transitive dep already present in the env). PURE/testable
        — no mlx involvement.
        """
        import yaml

        cfg_path = Path(path)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.build_config(), fh, sort_keys=False)
        return cfg_path

    @staticmethod
    def build_argv(config_path: str | Path) -> list[str]:
        """Build the ``mlx_lm.lora`` argv for the given YAML config path.

        Shape: ``["mlx_lm.lora", "--train", "--config", <path>]``. PURE/testable.
        The training hyperparameters all live in the YAML (``--config``) so the
        argv stays minimal and version-stable.
        """
        return ["mlx_lm.lora", "--train", "--config", str(config_path)]

    def run(self) -> MLXRunResult:
        """Execute ``mlx_lm.lora`` via the subprocess seam. The ONE un-runnable call.

        Behavior:

        * If ``check_feature("mlx")`` is False → raise
          :class:`~backpropagate.exceptions.MLXUnavailableError`
          (``code='DEP_MLX_UNAVAILABLE'``). mlx-lm is Apple-Silicon-only, so on
          this Windows/CUDA rig this is the path that fires; the Trainer
          constructor's forced-mlx-on-non-Apple guard normally intercepts first.
        * Otherwise write the YAML config, build the argv, and run it through
          :func:`backpropagate.export._run_subprocess_interruptible` (the SAME
          interrupt-safe seam the GGUF export uses). A non-zero exit
          (``CalledProcessError``) is wrapped into a structured
          :class:`~backpropagate.exceptions.TrainingError`
          (``code='RUNTIME_TRAINING_FAILED'``).
        * Final train/val loss are parsed best-effort from stdout; a parse miss
          yields ``final_loss=None`` + a WARN (never a failure).

        There is NO ``import mlx_lm`` here — the toolchain is invoked purely as a
        subprocess, so this module imports cleanly on a host where mlx is absent.
        """
        import subprocess

        from .exceptions import MLXUnavailableError, TrainingError

        if not check_feature("mlx"):
            raise MLXUnavailableError(
                reason="check_feature('mlx') is False (mlx_lm not importable).",
            )

        # Subprocess seam — imported lazily so a non-Apple host that only ever
        # builds configs never pays the export-module import cost here.
        from .export import _run_subprocess_interruptible

        adapter_path = Path(self.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        config_path = self.write_config(adapter_path / "lora_config.yaml")
        argv = self.build_argv(config_path)

        logger.info("mlx_lm.lora: launching %s", " ".join(argv))
        try:
            completed = _run_subprocess_interruptible(
                argv,
                timeout=self.timeout,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr_tail = (exc.stderr or "")[-2000:]
            raise TrainingError(
                f"mlx_lm.lora exited with code {exc.returncode}.",
                details={
                    "returncode": exc.returncode,
                    "argv": argv,
                    "stderr_tail": stderr_tail,
                },
                suggestion=(
                    "Inspect the mlx_lm.lora stderr above. Confirm the model id "
                    "is an mlx-community / HF model mlx-lm can load, the data "
                    "dir has a valid train.jsonl, and mlx-lm is current "
                    "(pip install -U 'backpropagate[mlx]')."
                ),
                code="RUNTIME_TRAINING_FAILED",
                cause=exc,
            ) from exc

        stdout = completed.stdout or ""
        final_loss, val_loss = _parse_final_losses(stdout)
        if final_loss is None:
            logger.warning(
                "mlx_lm.lora: could not parse a final train loss from stdout "
                "(mlx-lm log format may have drifted); recording final_loss=None. "
                "The adapter was still written to %s.",
                adapter_path,
            )

        return MLXRunResult(
            adapter_path=str(adapter_path),
            final_loss=final_loss,
            iters=int(self.iters),
            raw_stdout=stdout,
            val_loss=val_loss,
        )

    def fuse(
        self,
        save_path: str | Path,
        *,
        export_gguf: bool = False,
        gguf_path: str | Path | None = None,
    ) -> str:
        """Merge the trained adapter into the base weights via ``mlx_lm.fuse``.

        Thin wrapper over the SAME subprocess seam :meth:`run` uses (NO
        ``import mlx_lm``). Builds
        ``["mlx_lm.fuse", "--model", model, "--adapter-path", adapter_path,
        "--save-path", save_path]`` and, when ``export_gguf`` is set, appends
        ``--export-gguf [--gguf-path <gguf_path>]``.

        NB: mlx_lm.fuse's GGUF export is limited to Llama / Mistral-family fp16
        models (mlx-lm constraint). The fused safetensors directory otherwise
        feeds the existing ``export_ollama_adapter`` path unchanged.

        Returns the ``save_path`` as a string.
        """
        import subprocess

        from .exceptions import MLXUnavailableError, TrainingError

        if not check_feature("mlx"):
            raise MLXUnavailableError(
                reason="check_feature('mlx') is False (mlx_lm not importable).",
            )

        from .export import _run_subprocess_interruptible

        save = Path(save_path)
        save.mkdir(parents=True, exist_ok=True)
        argv = [
            "mlx_lm.fuse",
            "--model",
            str(self.model),
            "--adapter-path",
            str(self.adapter_path),
            "--save-path",
            str(save),
        ]
        if export_gguf:
            argv.append("--export-gguf")
            if gguf_path is not None:
                argv.extend(["--gguf-path", str(gguf_path)])

        logger.info("mlx_lm.fuse: launching %s", " ".join(argv))
        try:
            _run_subprocess_interruptible(
                argv,
                timeout=self.timeout,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise TrainingError(
                f"mlx_lm.fuse exited with code {exc.returncode}.",
                details={"returncode": exc.returncode, "argv": argv},
                suggestion=(
                    "Inspect the mlx_lm.fuse stderr. GGUF export only supports "
                    "Llama / Mistral-family fp16 models; the fused safetensors "
                    "dir can still feed export_ollama_adapter."
                ),
                code="RUNTIME_MERGE_EXPORT_FAILED",
                cause=exc,
            ) from exc

        return str(save)
