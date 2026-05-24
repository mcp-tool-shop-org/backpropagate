"""
Backpropagate - Model Export
============================

Export fine-tuned models to various formats.

Formats:
- LoRA adapter (default)
- Merged model (base + adapter)
- GGUF (for llama.cpp, Ollama, LM Studio)

GGUF Quantizations (fastest → smallest):
- f16: Full precision (largest)
- q8_0: 8-bit (best quality)
- q4_k_m: 4-bit (recommended balance)
- q4_0: 4-bit (fastest, lower quality)
"""

import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import (
    ExportError,
    GGUFExportError,
    InvalidSettingError,
    MergeExportError,
    OllamaRegistrationError,
)

# =============================================================================
# INPUT VALIDATORS (BRIDGE-A-001, BRIDGE-A-002)
# =============================================================================
# Both ``register_with_ollama`` and ``push_to_hub`` previously forwarded
# operator-supplied strings (the Ollama model name, the HF repo_id) into
# subprocess argv / network calls without validation. While ``shell=False``
# prevented shell-metachar injection on the ollama path, a leading ``-`` or
# embedded newline can still confuse the downstream CLI / HTTP library and
# produce option-injection or malformed-request surprises. We tighten the
# input contract here so the error message is structured (``ExportError`` with
# a stable ``code``) instead of leaking out of a downstream library.

# Ollama model name: typical shape is "user/name:tag" but we accept any name
# made up of alphanumerics + ``.``, ``_``, ``-``, ``:``. We disallow leading
# ``-`` (option injection) and any whitespace / control char (newline,
# carriage return, NUL would otherwise produce a malformed Modelfile or
# truncate the ``ollama create`` argv). Length cap 128 chars matches Ollama's
# own internal limit comfortably.
_OLLAMA_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")

# Hugging Face Hub repo_id: spec is ``<owner>/<name>`` where each segment
# matches ``[A-Za-z0-9][A-Za-z0-9._-]{0,95}``. We additionally reject any
# ``..`` segment (path traversal in cached README mirroring on Windows) and
# any control character.
_HF_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,95}/[A-Za-z0-9][A-Za-z0-9._\-]{0,95}$"
)


def _validate_model_name(model_name: str) -> None:
    """Reject Ollama model names that could confuse the ollama CLI parser.

    Raises:
        ExportError: with ``code='INPUT_VALIDATION_FAILED'`` when the name
            contains a control char, leading dash, whitespace, or otherwise
            fails the allowlist. The message is structured so the operator
            sees a clean CLI error instead of an obscure ``ollama create``
            failure.
    """
    if not isinstance(model_name, str) or not model_name:
        err = ExportError(
            "Ollama model name must be a non-empty string",
            suggestion="Pass --ollama-name <name> with a name like 'my-finetune' or 'org/model:tag'.",
            code="INPUT_VALIDATION_FAILED",
        )
        raise err

    # Reject NUL, newline, carriage return explicitly so the error message
    # names the offending character class (the regex below would catch them
    # via the allowlist, but the operator gets a clearer hint here).
    if any(ch in model_name for ch in ("\x00", "\n", "\r")):
        err = ExportError(
            "Ollama model name contains a control character (NUL / newline / CR)",
            suggestion="Strip the embedded control char from --ollama-name.",
            code="INPUT_VALIDATION_FAILED",
        )
        raise err

    if model_name.startswith("-"):
        err = ExportError(
            f"Ollama model name '{model_name}' starts with '-' — refusing to pass to ollama CLI",
            suggestion="A leading dash is interpreted as a flag by argv parsers. Rename without the dash.",
            code="INPUT_VALIDATION_FAILED",
        )
        raise err

    if not _OLLAMA_MODEL_NAME_RE.match(model_name):
        err = ExportError(
            f"Ollama model name '{model_name}' contains invalid characters",
            suggestion=(
                "Allowed: ASCII letters, digits, '.', '_', '-', ':' "
                "(first char must be alphanumeric); max 128 chars."
            ),
            code="INPUT_VALIDATION_FAILED",
        )
        raise err

    # Defense in depth: reject if Path(model_name).name differs (catches a
    # path separator that slipped past the regex on a future relax).
    if Path(model_name).name != model_name:
        err = ExportError(
            f"Ollama model name '{model_name}' contains path separators",
            suggestion="Model names cannot contain '/' or '\\'.",
            code="INPUT_VALIDATION_FAILED",
        )
        raise err


def _validate_repo_id(repo_id: str) -> None:
    """Reject malformed HF Hub repo identifiers BEFORE the network call.

    Raises:
        ExportError: with ``code='HUB_PUSH_INVALID_REPO'`` when the value is
            empty, missing the ``owner/name`` shape, contains ``..``
            segments, leading dash on either segment, or control chars.
    """
    if not isinstance(repo_id, str) or not repo_id:
        err = ExportError(
            "Hugging Face repo_id must be a non-empty string",
            suggestion="Pass --repo owner/name (e.g. 'alice/qwen-finetune').",
        )
        err.code = "HUB_PUSH_INVALID_REPO"  # type: ignore[attr-defined]
        raise err

    if any(ch in repo_id for ch in ("\x00", "\n", "\r", "\\")):
        err = ExportError(
            f"repo_id '{repo_id}' contains a control character or backslash",
            suggestion="Strip embedded control chars / backslashes from --repo.",
        )
        err.code = "HUB_PUSH_INVALID_REPO"  # type: ignore[attr-defined]
        raise err

    # Explicit '..' check before the regex so the message can name traversal.
    segments = repo_id.split("/")
    if any(seg == ".." or seg == "." for seg in segments):
        err = ExportError(
            f"repo_id '{repo_id}' contains a '..' or '.' segment",
            suggestion="Use the 'owner/name' shape with no relative-path segments.",
        )
        err.code = "HUB_PUSH_INVALID_REPO"  # type: ignore[attr-defined]
        raise err

    if not _HF_REPO_ID_RE.match(repo_id):
        err = ExportError(
            f"repo_id '{repo_id}' is not in the 'owner/name' shape required by Hugging Face",
            suggestion=(
                "Each segment must start alphanumeric and use only "
                "[A-Za-z0-9._-]; max 96 chars per segment."
            ),
        )
        err.code = "HUB_PUSH_INVALID_REPO"  # type: ignore[attr-defined]
        raise err


def _run_subprocess_interruptible(
    cmd: list[str],
    *,
    timeout: float,
    capture_output: bool = True,
    text: bool = True,
    check: bool = True,
    **popen_kwargs: Any,
) -> "subprocess.CompletedProcess[str]":
    """Run ``cmd`` with KeyboardInterrupt that actually propagates to the child.

    BRIDGE-A-004: ``subprocess.run(cmd, timeout=N)`` does NOT reliably forward
    SIGINT / Ctrl+C to the spawned child on Windows (a console-CTRL_C is
    multiplexed across the whole process group; the child may swallow it) or
    on POSIX (the child may be re-parented before the parent's KeyboardInterrupt
    handler fires). The result is zombie quantization processes that hold
    GPU + disk for many minutes after the operator pressed Ctrl+C.

    We replace ``subprocess.run`` with an explicit ``Popen`` whose child runs
    in its OWN process group / session, then ``.wait(timeout=...)`` on the
    parent side. KeyboardInterrupt at the parent terminates the child first
    (SIGTERM on POSIX, CTRL_BREAK_EVENT on Windows), waits up to 10s for
    cleanup, then re-raises so the caller's ``except KeyboardInterrupt``
    branch can run as before.

    Returns a ``CompletedProcess`` shaped exactly like ``subprocess.run`` for
    drop-in replacement at the existing raise-site.
    """
    # sys.platform-based check (mypy narrows these as platform-conditional;
    # os.name == "nt" does NOT narrow, so we'd hit attr-defined errors on
    # Linux/macOS mypy runs even though the branch never executes there).
    if sys.platform == "win32":
        popen_kwargs.setdefault("creationflags", subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        popen_kwargs.setdefault("start_new_session", True)

    stdout_dest = subprocess.PIPE if capture_output else None
    stderr_dest = subprocess.PIPE if capture_output else None

    proc = subprocess.Popen(  # noqa: S603 — argv is callee-controlled here
        cmd,
        stdout=stdout_dest,
        stderr=stderr_dest,
        text=text,
        **popen_kwargs,
    )

    try:
        stdout_data, stderr_data = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Tear down the child explicitly so it doesn't outlive the parent.
        try:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            proc.kill()
        raise
    except KeyboardInterrupt:
        # Propagate Ctrl+C to the child (it didn't get its own console signal
        # because we put it in a separate group/session) so it actually stops
        # writing instead of leaving a zombie.
        try:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            proc.kill()
        raise

    completed: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout_data if capture_output else None,
        stderr=stderr_data if capture_output else None,
    )

    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=stdout_data, stderr=stderr_data
        )

    return completed

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


__all__ = [
    "GGUFQuantization",
    "ExportFormat",
    "ExportResult",
    "export_lora",
    "export_merged",
    "export_gguf",
    "create_modelfile",
    "register_with_ollama",
    "list_ollama_models",
    # F-001 / F-004
    "push_to_hub",
    "write_model_card",
]


def _maybe_write_model_card(
    output_dir: str | Path,
    *,
    enabled: bool,
    run_id: str | None,
    base_model: str | None,
    output_root: str | Path | None,
    extra_card_fields: dict[str, Any] | None = None,
) -> Path | None:
    """F-004 helper: write model_card.md alongside an export when enabled.

    Pulls metadata from the on-disk RunHistoryManager record if available
    so the card reflects what actually happened during training. Falls
    back to the explicit ``extra_card_fields`` (and ``incomplete_provenance``)
    when no run is found. Returns the path written, or None when disabled
    or on failure (errors logged at WARN — model-card emission must never
    kill a successful export).
    """
    if not enabled:
        return None

    from .model_card import load_run_history_for_card, write_model_card_for_export

    output_dir_path = Path(output_dir)
    # The card lives next to model artifacts when output_dir is a directory;
    # for the GGUF single-file case the caller passes the parent directory.
    if output_dir_path.is_file():
        output_dir_path = output_dir_path.parent

    run_record: dict[str, Any] | None = None
    candidate_roots: list[str | Path] = []
    if output_root is not None:
        candidate_roots.append(output_root)
    # Also try walking up from the export directory: typical layout is
    # <output>/<format>/, so <output> sits one level up from output_dir_path.
    candidate_roots.append(output_dir_path)
    candidate_roots.append(output_dir_path.parent)

    for root in candidate_roots:
        run_record = load_run_history_for_card(root, run_id)
        if run_record:
            break

    fields: dict[str, Any] = {
        "run_id": run_id,
        "base_model": base_model,
        "incomplete_provenance": run_record is None,
    }

    if run_record:
        hp = run_record.get("hyperparameters") or {}
        fields.update({
            "base_model": run_record.get("model_name") or base_model,
            "dataset_path": run_record.get("dataset_info"),
            "dataset_hash": run_record.get("dataset_hash"),
            "final_loss": run_record.get("final_loss"),
            "loss_history": run_record.get("loss_history") or [],
            "steps": run_record.get("steps"),
            "lora_r": hp.get("lora_r"),
            "lora_alpha": hp.get("lora_alpha"),
            "seed": hp.get("seed"),
            "training_duration": run_record.get("duration_seconds"),
        })

    if extra_card_fields:
        for key, value in extra_card_fields.items():
            if value is not None:
                fields[key] = value

    try:
        return write_model_card_for_export(output_dir_path, **fields)
    except (OSError, PermissionError) as exc:
        logger.warning(f"model card emission failed: {exc}")
        return None


def write_model_card(
    output_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """F-004 thin wrapper around :func:`backpropagate.model_card.write_model_card_for_export`.

    Re-exported here so operators using the export surface don't need to
    cross the package boundary into ``backpropagate.model_card`` directly.
    """
    from .model_card import write_model_card_for_export

    return write_model_card_for_export(output_dir, **kwargs)


# =============================================================================
# F-001 HUGGING FACE HUB PUSH
# =============================================================================
# Operators want a one-step "publish my adapter / merged model to the Hub"
# command. We keep the surface narrow and explicit:
#   * ``push_to_hub(local_path, repo_id, ...)`` uploads a directory or
#     single file to ``repo_id`` via :class:`huggingface_hub.HfApi`.
#   * The accompanying ``model_card.md`` (F-004) is renamed to ``README.md``
#     inside the upload so HF's UI picks it up as the repo card.
#   * Errors are wrapped with cause_category-aware messages so the operator
#     sees the right hint (auth / not_found / network / version / unknown).


def _resolve_hf_token(token: str | None) -> str | None:
    """Resolve the HF token to pass to HfApi.

    Resolution order: explicit arg → ``HF_TOKEN`` env var →
    ``HUGGING_FACE_HUB_TOKEN`` env var → token cached by
    ``huggingface-cli login`` at ``~/.cache/huggingface/token``. Returns
    ``None`` when none of these are set; HfApi will then fall back to
    whatever the underlying ``huggingface_hub`` session was configured
    with (e.g. an active ``HF_HUB_TOKEN`` from a Colab integration).
    """

    if token:
        return token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token
    cache_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token_path.exists():
        try:
            cached = cache_token_path.read_text(encoding="utf-8").strip()
            if cached:
                return cached
        except OSError:
            pass
    return None


def push_to_hub(
    local_path: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    create_repo: bool = True,
    repo_type: str = "model",
    include_base: bool = False,
) -> str:
    """F-001: upload a local adapter / merged model to the Hugging Face Hub.

    Args:
        local_path: Directory (or single file) to upload.
        repo_id: HF Hub repo identifier, e.g. ``"alice/qwen-finetune"``.
        token: HF token. Falls back to ``HF_TOKEN``,
            ``HUGGING_FACE_HUB_TOKEN``, or the cached
            ``~/.cache/huggingface/token`` from ``huggingface-cli login``.
        private: When True, create the repo as private.
        commit_message: Commit message for this upload (default:
            ``"Upload via backpropagate"``).
        create_repo: When True (default), create the repo if it doesn't
            yet exist. When False, expect the repo to be present and
            error out with a clear hint if not.
        repo_type: ``"model"`` (default) | ``"dataset"`` | ``"space"``.
        include_base: When the upload directory is an adapter-only LoRA
            export, default behaviour is to upload only the adapter
            files. Set ``True`` to push every file in the directory
            (including the base model if it happens to be there).

    Returns:
        The Hub URL for the published repo (``https://huggingface.co/<repo_id>``).

    Raises:
        ExportError: For auth / not_found / network / version failures.
            ``ExportError.code`` is set to ``HUB_PUSH_AUTH`` /
            ``HUB_PUSH_NOT_FOUND`` / ``HUB_PUSH_NETWORK`` /
            ``HUB_PUSH_VERSION`` / ``HUB_PUSH_UNKNOWN`` so callers can
            distinguish.
    """
    # BRIDGE-A-002: validate repo_id BEFORE the imports so a malformed value
    # produces a structured error even when huggingface_hub isn't installed.
    # ExportError.code='HUB_PUSH_INVALID_REPO' makes the failure scrapeable.
    _validate_repo_id(repo_id)

    try:
        from huggingface_hub import HfApi  # type: ignore
        from huggingface_hub import create_repo as hf_create_repo  # type: ignore
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore
    except ImportError as exc:
        raise ExportError(
            "huggingface_hub is not installed",
            suggestion=(
                "Install with `pip install huggingface_hub` (typically pulled "
                "in by `pip install backpropagate[unsloth]`)."
            ),
        ) from exc

    local_path_p = Path(local_path).expanduser().resolve()
    if not local_path_p.exists():
        raise ExportError(
            f"Local path does not exist: {local_path_p}",
            suggestion="Pass a directory produced by `backprop export` or `backprop train --output`.",
        )

    resolved_token = _resolve_hf_token(token)
    api = HfApi(token=resolved_token)
    commit_message = commit_message or "Upload via backpropagate"

    # If a model_card.md sits next to the artifacts, mirror it as README.md
    # so the HF web UI renders it as the repo's model card.
    # BRIDGE-A-008 (Stage C amend): defensive checks BEFORE read_text.
    #   1. Refuse to mirror a symlinked model_card.md — on POSIX a malicious
    #      or accidental symlink to ``~/.ssh/id_rsa`` / ``/etc/passwd`` would
    #      otherwise be uploaded verbatim to the public HF repo as README.md.
    #      The check is a no-op on Windows where regular symlinks require
    #      admin to create.
    #   2. Cap the mirrored file at 1 MB. HF model cards are typically
    #      <50 KB; a 50 MB tampered or accidentally-bloated card would
    #      otherwise balloon the upload and push the operator's HF storage
    #      quota for no usability benefit.
    # Failures here log + skip the mirror — the export itself continues.
    _MAX_MODEL_CARD_BYTES = 1_000_000
    readme_copy_path: Path | None = None
    if local_path_p.is_dir():
        candidate_card = local_path_p / "model_card.md"
        readme_target = local_path_p / "README.md"
        if candidate_card.exists() and not readme_target.exists():
            if candidate_card.is_symlink():
                logger.warning(
                    "Refusing to mirror symlinked model_card.md to README.md: "
                    f"{candidate_card} (would leak the symlink target to the Hub)"
                )
            else:
                try:
                    card_size = candidate_card.stat().st_size
                except OSError as exc:
                    logger.warning(
                        f"Could not stat model_card.md at {candidate_card}: {exc}"
                    )
                    card_size = None
                if card_size is not None and card_size > _MAX_MODEL_CARD_BYTES:
                    logger.warning(
                        f"Refusing to mirror model_card.md to README.md: "
                        f"file is {card_size} bytes (cap "
                        f"{_MAX_MODEL_CARD_BYTES}). HF model cards are "
                        "typically <50 KB."
                    )
                elif card_size is not None:
                    try:
                        readme_target.write_text(
                            candidate_card.read_text(encoding="utf-8"),
                            encoding="utf-8",
                        )
                        readme_copy_path = readme_target
                        logger.info(
                            "Mirrored model_card.md to README.md for the Hub upload "
                            f"at {readme_target}"
                        )
                    except OSError as exc:
                        logger.warning(
                            f"Could not mirror model_card.md to README.md: {exc}"
                        )

    # BRIDGE-A-010 (Stage C amend): track whether the upload itself failed so
    # the finally block can actually act on the docstring promise ("remove
    # only if we created it and the upload itself failed"). Pre-fix, the
    # finally's body was a no-op tested-for-non-existence — meaning a failed
    # push left the local README.md mirror in place with stale content, and
    # the next attempt found it (via the `and not readme_target.exists()`
    # guard above) and skipped re-creation. The new flag is set in every
    # exception branch; on success it stays False and the mirror is kept
    # (the documented "keep so local matches Hub" behavior).
    upload_failed = False
    # BRIDGE-B-007 (Stage C humanization): instrument each push_to_hub stage
    # so a JSON-log consumer can see (a) when the push started, (b) what
    # files / bytes are being uploaded, (c) when the repo was created
    # (vs. reused), and (d) duration on success. The "==> Pushing to Hub"
    # print at cli.py:866 is the only user-facing signal pre-fix; for a
    # 4-minute 7B merged-model push there is no progress event in the
    # structured log stream. Imports stay lazy to avoid pulling structlog
    # into the cold path of `backprop --help`.
    import time as _push_time
    _push_started_at = _push_time.monotonic()

    # Inventory the upload payload so the started event names file_count +
    # total_bytes — operators triaging a stuck push want to know if they
    # are 30 seconds into a 7B-model upload or 30 seconds into a 50-MB
    # adapter-only push.
    def _inventory(path: Path) -> tuple[int, int]:
        if path.is_file():
            try:
                return 1, path.stat().st_size
            except OSError:
                return 1, 0
        total_files = 0
        total_bytes = 0
        for f in path.rglob("*"):
            if f.is_file():
                total_files += 1
                try:
                    total_bytes += f.stat().st_size
                except OSError:
                    pass
        return total_files, total_bytes

    _file_count, _total_bytes = _inventory(local_path_p)
    try:
        logger.info(
            "hub_push_started repo_id=%s local_path=%s file_count=%d total_bytes=%d private=%s repo_type=%s",
            repo_id,
            str(local_path_p),
            _file_count,
            _total_bytes,
            private,
            repo_type,
        )
    except Exception:  # noqa: BLE001 — observability must not block the push  # nosec B110
        pass

    try:
        if create_repo:
            hf_create_repo(
                repo_id,
                token=resolved_token,
                private=private,
                exist_ok=True,
                repo_type=repo_type,
            )
            try:
                logger.info(
                    "hub_push_repo_ready repo_id=%s private=%s exist_ok=True",
                    repo_id,
                    private,
                )
            except Exception:  # noqa: BLE001  # nosec B110
                pass

        # Filter set for adapter-only push. We default to uploading only
        # ``adapter_*`` / ``model_card.md`` / ``README.md`` / config files
        # so an export of a quantized base model doesn't accidentally get
        # rebroadcast through the operator's Hub upload.
        allow_patterns: list[str] | None = None
        if local_path_p.is_dir() and not include_base:
            adapter_files = list(local_path_p.glob("adapter_*"))
            if adapter_files:
                allow_patterns = [
                    "adapter_*",
                    "*.json",
                    "*.md",
                    "tokenizer*",
                    "special_tokens_map.json",
                    "vocab*",
                    "merges*",
                ]

        if local_path_p.is_file():
            api.upload_file(
                path_or_fileobj=str(local_path_p),
                path_in_repo=local_path_p.name,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                token=resolved_token,
            )
        else:
            api.upload_folder(
                folder_path=str(local_path_p),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                allow_patterns=allow_patterns,
                token=resolved_token,
            )

    except HfHubHTTPError as exc:
        # BRIDGE-A-010 (Stage C amend): mark upload_failed once at the top of
        # the HfHubHTTPError branch — every sub-status (401/403/404/5xx)
        # represents a failed upload, so the README mirror should be rolled
        # back uniformly via the finally clause below.
        upload_failed = True
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            # BRIDGE-A-007 (catalog drift fix): use the canonical
            # INPUT_AUTH_REQUIRED code that already exists in ERROR_CODES
            # rather than the orphaned HUB_PUSH_AUTH string. The CLI branch
            # below tests both for back-compat during the rename.
            err = ExportError(
                f"Hugging Face authentication failed (HTTP {status}): {exc}",
                suggestion=(
                    "Pass --token or set HF_TOKEN to a token with write "
                    "access to the target repo. Run `huggingface-cli login` "
                    "for a stored token."
                ),
            )
            err.code = "INPUT_AUTH_REQUIRED"  # type: ignore[attr-defined]
            raise err from exc
        if status == 404:
            err = ExportError(
                f"Hugging Face repo not found (HTTP {status}): {exc}",
                suggestion=(
                    "Verify repo_id is spelled correctly. Pass "
                    "--include-base or omit --create-repo=false to let "
                    "backpropagate create the repo for you."
                ),
            )
            err.code = "HUB_PUSH_NOT_FOUND"  # type: ignore[attr-defined]
            raise err from exc
        err = ExportError(
            f"Hugging Face Hub push failed (HTTP {status}): {exc}",
            suggestion=(
                "Retry — the Hub occasionally returns 5xx for ~30s "
                "during peak load. Verify your network if the error "
                "persists."
            ),
        )
        err.code = "HUB_PUSH_NETWORK" if status and status >= 500 else "HUB_PUSH_UNKNOWN"  # type: ignore[attr-defined]
        raise err from exc
    except (ConnectionError, TimeoutError) as exc:
        upload_failed = True
        err = ExportError(
            f"Network error contacting Hugging Face Hub: {exc}",
            suggestion="Check your network and retry.",
        )
        err.code = "HUB_PUSH_NETWORK"  # type: ignore[attr-defined]
        raise err from exc
    except ExportError:
        # Pre-upload validation errors (e.g. _validate_repo_id) reach here.
        # We did NOT touch the network and the README mirror — if any — was
        # written above, so don't mark upload_failed (keep the mirror so the
        # operator can inspect it). Re-raise unchanged.
        raise
    except Exception as exc:
        upload_failed = True
        err = ExportError(
            f"Hugging Face Hub push failed: {exc}",
            suggestion=(
                "Retry. If the failure persists, run with --verbose to see "
                "the full traceback."
            ),
        )
        err.code = "HUB_PUSH_UNKNOWN"  # type: ignore[attr-defined]
        raise err from exc
    finally:
        # BRIDGE-A-010 (Stage C amend): the docstring promises "remove only
        # if we created it and the upload itself failed". Pre-fix this was a
        # no-op (tested for NON-existence and did nothing). Now: when we
        # actually created the README mirror AND the upload failed, unlink
        # so the next attempt re-mirrors fresh model_card.md content instead
        # of finding the stale mirror and skipping. On success the mirror is
        # kept so the operator's local dir matches the Hub.
        if upload_failed and readme_copy_path is not None:
            try:
                readme_copy_path.unlink(missing_ok=True)
                logger.debug(
                    f"Rolled back mirrored README.md at {readme_copy_path} "
                    "after upload failure"
                )
            except OSError as cleanup_exc:
                # Cleanup failure must not mask the real exception from the
                # except branches above — log + swallow so the original
                # HfHubHTTPError / ConnectionError surfaces unchanged.
                logger.debug(
                    f"README rollback cleanup failed at {readme_copy_path}: "
                    f"{cleanup_exc}"
                )

    # BRIDGE-B-007 (Stage C): hub_push_complete fires on the success path
    # only (the exception branches above already mark upload_failed=True
    # and re-raise with HUB_PUSH_* codes that operators can grep). The
    # duration is rounded to 0.1s so the JSON log stays compact.
    try:
        _duration = _push_time.monotonic() - _push_started_at
        logger.info(
            "hub_push_complete repo_id=%s url=https://huggingface.co/%s duration_seconds=%.1f file_count=%d total_bytes=%d",
            repo_id,
            repo_id,
            _duration,
            _file_count,
            _total_bytes,
        )
    except Exception:  # noqa: BLE001  # nosec B110
        pass

    return f"https://huggingface.co/{repo_id}"


class GGUFQuantization(Enum):
    """GGUF quantization levels (fastest → smallest)."""

    F16 = "f16"
    Q8_0 = "q8_0"
    Q5_K_M = "q5_k_m"
    Q4_K_M = "q4_k_m"
    Q4_0 = "q4_0"
    Q2_K = "q2_k"


class ExportFormat(Enum):
    """Model export formats."""

    LORA = "lora"
    MERGED = "merged"
    GGUF = "gguf"


@dataclass
class ExportResult:
    """Result of a model export operation."""

    format: ExportFormat
    path: Path
    size_mb: float
    quantization: str | None = None  # For GGUF
    export_time_seconds: float = 0.0

    def summary(self) -> str:
        """Human-readable summary of the export."""
        lines = [
            "Export Complete",
            f"  Format: {self.format.value}",
            f"  Path: {self.path}",
            f"  Size: {self.size_mb:.1f} MB",
        ]
        if self.quantization:
            lines.append(f"  Quantization: {self.quantization}")
        if self.export_time_seconds > 0:
            lines.append(f"  Time: {self.export_time_seconds:.1f}s")
        return "\n".join(lines)


def _get_dir_size_mb(path: Path) -> float:
    """Get total size of directory in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def _is_peft_model(model: Any) -> bool:
    """Check if model is a PeftModel."""
    try:
        from peft import PeftModel

        return isinstance(model, PeftModel)
    except ImportError:
        return False


def _has_unsloth() -> bool:
    """Check if Unsloth is available."""
    try:
        # Suppress import order warning - expected when checking availability
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Unsloth should be imported before.*")
            import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def export_lora(
    model: Any,
    output_dir: str | Path,
    adapter_name: str = "default",
    *,
    emit_model_card: bool = True,
    run_id: str | None = None,
    base_model: str | None = None,
    output_root: str | Path | None = None,
) -> ExportResult:
    """
    Export LoRA adapter only, atomically (B-006).

    Writes flow into ``<output_dir>.partial`` first and ``shutil.move()``
    promotes the directory to the final path on success. The partial
    directory is removed on any failure so a disk-full crash mid-write
    doesn't leave a half-populated adapter directory behind (which would
    raise a cryptic 'state_dict missing keys' on the next resume).

    F-004: when ``emit_model_card=True`` (the default) a ``model_card.md``
    is written into ``output_dir`` after the atomic promote. Provenance is
    pulled from the on-disk RunHistoryManager record at
    ``output_root/run_history.json`` (defaults to ``output_dir.parent``
    when not provided). Failures in model-card emission are logged but
    never abort the export.

    Args:
        model: PeftModel or path to saved model
        output_dir: Directory to save adapter
        adapter_name: Name of adapter to save
        emit_model_card: F-004 — emit a model_card.md alongside the export
            (default True). Pass False to opt out.
        run_id: Correlation token used to look up training metadata.
        base_model: HF identifier of the base model (used as a frontmatter
            field on the card when no RunHistoryManager record is found).
        output_root: Root output directory to search for run_history.json.

    Returns:
        ExportResult with path and size info

    Raises:
        ExportError: If export fails
    """
    start_time = time.time()
    output_path = Path(output_dir)
    partial_path = output_path.with_name(output_path.name + ".partial")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise ExportError(
            f"Cannot create parent directory: {e}",
            suggestion=f"Check write permissions for {output_path.parent}"
        ) from e
    except OSError as e:
        raise ExportError(f"Failed to create parent directory: {e}") from e

    if partial_path.exists():
        shutil.rmtree(partial_path, ignore_errors=True)

    try:
        partial_path.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        raise ExportError(f"Failed to create partial directory: {e}") from e

    try:
        if isinstance(model, (str, Path)):
            # Copy existing adapter
            src_path = Path(model)
            if not src_path.exists():
                raise ExportError(
                    f"Source model path does not exist: {src_path}",
                    suggestion="Check that the model was trained and saved correctly"
                )
            if src_path.is_dir():
                # Copy adapter files
                files_copied = 0
                for pattern in ["adapter_*.safetensors", "adapter_*.bin", "adapter_config.json"]:
                    for f in src_path.glob(pattern):
                        shutil.copy2(f, partial_path / f.name)
                        files_copied += 1
                if files_copied == 0:
                    raise ExportError(
                        f"No adapter files found in {src_path}",
                        suggestion="Ensure the directory contains adapter_*.safetensors or adapter_*.bin files"
                    )
        elif _is_peft_model(model):
            # Save from PeftModel
            model.save_pretrained(partial_path, adapter_name=adapter_name)
        else:
            raise ExportError(
                f"Cannot export LoRA from {type(model).__name__}",
                suggestion="Expected PeftModel or path to saved adapter"
            )

        # Atomic promote.
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(partial_path), str(output_path))
    except ExportError:
        raise
    except Exception as e:
        raise ExportError(f"LoRA export failed: {e}") from e
    finally:
        # Belt-and-braces cleanup on every path.
        if partial_path.exists():
            shutil.rmtree(partial_path, ignore_errors=True)

    export_time = time.time() - start_time
    size_mb = _get_dir_size_mb(output_path)

    logger.info(f"LoRA adapter exported to {output_path} ({size_mb:.1f} MB)")

    # F-004: emit model card alongside the artifact.
    _maybe_write_model_card(
        output_path,
        enabled=emit_model_card,
        run_id=run_id,
        base_model=base_model,
        output_root=output_root,
        extra_card_fields={"export_format": "lora"},
    )

    return ExportResult(
        format=ExportFormat.LORA,
        path=output_path,
        size_mb=size_mb,
        export_time_seconds=export_time,
    )


def export_merged(
    model: Any,
    tokenizer: "PreTrainedTokenizer",
    output_dir: str | Path,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    *,
    emit_model_card: bool = True,
    run_id: str | None = None,
    base_model: str | None = None,
    output_root: str | Path | None = None,
) -> ExportResult:
    """
    Merge adapter into base model and save.

    F-004: emits ``model_card.md`` alongside the merged checkpoint when
    ``emit_model_card=True`` (default).

    Args:
        model: PeftModel with adapter
        tokenizer: Tokenizer to save with model
        output_dir: Directory to save merged model
        push_to_hub: Whether to push to Hugging Face Hub
        repo_id: Repository ID for Hub (required if push_to_hub=True)
        emit_model_card: F-004 — emit model_card.md alongside the export.
        run_id: Correlation token used to look up training metadata.
        base_model: Base model HF identifier (frontmatter fallback).
        output_root: Root output directory to search for run_history.json.

    Returns:
        ExportResult with path and size info

    Raises:
        MergeExportError: If merge or save fails
    """
    start_time = time.time()
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise MergeExportError(
            f"Cannot create output directory: {e}",
            suggestion=f"Check write permissions for {output_path.parent}"
        ) from e
    except OSError as e:
        raise MergeExportError(f"Failed to create output directory: {e}") from e

    if not _is_peft_model(model):
        raise MergeExportError(
            f"Cannot merge non-PeftModel (got {type(model).__name__})",
            suggestion="Ensure you're exporting a model with LoRA adapters applied"
        )

    try:
        # Merge and unload adapter
        merged_model = model.merge_and_unload()
    except Exception as e:
        raise MergeExportError(f"Failed to merge LoRA adapters: {e}") from e

    try:
        # Save model and tokenizer
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        raise MergeExportError(f"Failed to save merged model: {e}") from e

    # Push to Hub if requested
    if push_to_hub:
        if not repo_id:
            raise MergeExportError(
                "repo_id required when push_to_hub=True",
                suggestion="Provide repo_id='username/model-name'"
            )
        try:
            merged_model.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            logger.info(f"Pushed merged model to HuggingFace Hub: {repo_id}")
        except Exception as e:
            raise MergeExportError(
                f"Failed to push to HuggingFace Hub: {e}",
                suggestion="Check your HuggingFace token and repo_id"
            ) from e

    export_time = time.time() - start_time
    size_mb = _get_dir_size_mb(output_path)

    logger.info(f"Merged model exported to {output_path} ({size_mb:.1f} MB)")

    # F-004: model_card.md
    _maybe_write_model_card(
        output_path,
        enabled=emit_model_card,
        run_id=run_id,
        base_model=base_model,
        output_root=output_root,
        extra_card_fields={"export_format": "merged"},
    )

    return ExportResult(
        format=ExportFormat.MERGED,
        path=output_path,
        size_mb=size_mb,
        export_time_seconds=export_time,
    )


def export_gguf(
    model: Any,
    tokenizer: "PreTrainedTokenizer",
    output_dir: str | Path,
    quantization: str | GGUFQuantization = "q4_k_m",
    model_name: str | None = None,
    *,
    emit_model_card: bool = True,
    run_id: str | None = None,
    base_model: str | None = None,
    output_root: str | Path | None = None,
) -> ExportResult:
    """
    Export to GGUF format.

    Uses Unsloth's save_pretrained_gguf if available (much faster).
    Falls back to manual conversion via llama.cpp if needed.

    Args:
        model: Model to export (PeftModel or base model)
        tokenizer: Tokenizer for the model
        output_dir: Directory to save GGUF file
        quantization: Quantization level (default: q4_k_m)
        model_name: Name for the output file (default: "model")

    Returns:
        ExportResult with path and size info

    Raises:
        GGUFExportError: If GGUF export fails
        InvalidSettingError: If quantization is invalid
    """
    start_time = time.time()
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise GGUFExportError(
            f"Cannot create output directory: {e}",
            output_path=str(output_path),
        ) from e
    except OSError as e:
        raise GGUFExportError(
            f"Failed to create output directory: {e}",
            output_path=str(output_path),
        ) from e

    # Normalize quantization
    if isinstance(quantization, GGUFQuantization):
        quant_str = quantization.value
    else:
        quant_str = quantization.lower()

    # Validate quantization
    valid_quants = {q.value for q in GGUFQuantization}
    if quant_str not in valid_quants:
        raise InvalidSettingError(
            "quantization",
            quant_str,
            f"one of {sorted(valid_quants)}",
            suggestion="Try 'q4_k_m' for a good balance of size and quality"
        )

    model_name = model_name or "model"

    # Try Unsloth first (fastest)
    # B-006: write into a sibling .partial directory so a mid-conversion
    # disk-full / crash doesn't leave a half-written .gguf file at the
    # final path. On success we move the produced .gguf into output_path.
    if _has_unsloth():
        unsloth_partial = output_path / "_unsloth_partial"
        if unsloth_partial.exists():
            shutil.rmtree(unsloth_partial, ignore_errors=True)
        try:
            unsloth_partial.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            raise GGUFExportError(
                f"Failed to create partial directory: {e}",
                output_path=str(output_path),
                quantization=quant_str,
            ) from e

        try:
            print("Exporting to GGUF format... this may take several minutes for large models.")
            # Unsloth handles everything (writes into the partial dir)
            model.save_pretrained_gguf(
                str(unsloth_partial),
                tokenizer,
                quantization_method=quant_str,
            )

            # Find the generated GGUF file in the partial dir.
            gguf_files = list(unsloth_partial.glob("*.gguf"))
            if not gguf_files:
                raise GGUFExportError(
                    f"GGUF file was not created in partial directory: {unsloth_partial}",
                    output_path=str(output_path),
                    quantization=quant_str,
                    suggestion="Check Unsloth logs for conversion errors"
                )

            # Atomic promote: move the final .gguf into output_path.
            final_gguf_name = gguf_files[0].name
            gguf_path = output_path / final_gguf_name
            if gguf_path.exists():
                gguf_path.unlink()
            shutil.move(str(gguf_files[0]), str(gguf_path))

            # Validate output exists at the final path
            if not gguf_path.exists():
                raise GGUFExportError(
                    f"GGUF file was not promoted to expected path: {gguf_path}",
                    output_path=str(output_path),
                    quantization=quant_str,
                    suggestion="Check Unsloth logs for conversion errors"
                )

            export_time = time.time() - start_time
            size_mb = _get_dir_size_mb(gguf_path)

            if size_mb == 0:
                raise GGUFExportError(
                    f"GGUF file is empty (0 bytes): {gguf_path}",
                    output_path=str(output_path),
                    quantization=quant_str,
                )

            logger.info(f"GGUF exported via Unsloth to {gguf_path} ({size_mb:.1f} MB)")

            # B-006: clean up the partial dir (any leftover Unsloth scratch
            # files that we didn't promote) before returning.
            if unsloth_partial.exists():
                shutil.rmtree(unsloth_partial, ignore_errors=True)

            # F-004: model card next to the GGUF (single-file export).
            _maybe_write_model_card(
                gguf_path,
                enabled=emit_model_card,
                run_id=run_id,
                base_model=base_model,
                output_root=output_root,
                extra_card_fields={
                    "export_format": "gguf",
                    "quantization": quant_str,
                },
            )

            return ExportResult(
                format=ExportFormat.GGUF,
                path=gguf_path,
                size_mb=size_mb,
                quantization=quant_str,
                export_time_seconds=export_time,
            )
        except GGUFExportError:
            # Clean up the partial dir on a structured failure too.
            if unsloth_partial.exists():
                shutil.rmtree(unsloth_partial, ignore_errors=True)
            raise
        except Exception as e:
            # Fall through to manual conversion. The partial dir is removed
            # so the next attempt has a clean slot.
            if unsloth_partial.exists():
                shutil.rmtree(unsloth_partial, ignore_errors=True)
            logger.warning(f"Unsloth GGUF export failed: {e}. Trying manual conversion...")
            print("WARNING: Unsloth GGUF export failed, falling back to llama.cpp conversion (slower)...")

    # Manual conversion: merge first, then convert
    # This requires llama.cpp's convert script
    merged_path = output_path / "merged_temp"

    try:
        merged_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise GGUFExportError(
            f"Failed to create temp directory for merge: {e}",
            output_path=str(output_path),
        ) from e

    try:
        # Merge if needed
        if _is_peft_model(model):
            merged_model = model.merge_and_unload()
        else:
            merged_model = model

        # Save in HF format
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
    except Exception as e:
        # Clean up on failure
        shutil.rmtree(merged_path, ignore_errors=True)
        raise GGUFExportError(
            f"Failed to prepare model for GGUF conversion: {e}",
            output_path=str(output_path),
        ) from e

    # Try to find llama.cpp convert script
    gguf_path = output_path / f"{model_name}-{quant_str}.gguf"

    # Check for llama-cpp-python or llama.cpp (system/user paths only, no CWD-relative)
    #
    # BRIDGE-B-014 (Stage C humanization): path discovery now honors:
    #   1. ``BACKPROPAGATE_LLAMA_CPP_PATH`` env var — operator escape hatch
    #      for non-standard install locations (CI runners under /opt, chocolatey
    #      under C:\tools, etc.). Value may be the script path or the
    #      llama.cpp directory containing it.
    #   2. ``shutil.which('convert_hf_to_gguf.py')`` — catches PATH-based
    #      installs on every OS (Linux pip, Windows scoop, macOS brew).
    #   3. ``~/llama.cpp/convert_hf_to_gguf.py`` (the original home-dir path).
    #   4. ``/usr/local/bin/convert_hf_to_gguf.py`` (the original POSIX
    #      system-wide path — meaningless on Windows but harmless).
    # The error suggestion below enumerates ALL paths searched so the operator
    # knows what to populate / what to set.
    import shutil as _shutil
    convert_script = None
    searched_paths: list[Path] = []

    env_override = os.environ.get("BACKPROPAGATE_LLAMA_CPP_PATH")
    if env_override:
        env_path = Path(env_override).expanduser()
        # Accept either a direct script path or a containing directory.
        if env_path.is_dir():
            env_path = env_path / "convert_hf_to_gguf.py"
        searched_paths.append(env_path)
        if env_path.exists():
            convert_script = env_path

    if convert_script is None:
        # shutil.which returns the first match in PATH; works cross-OS.
        which_result = _shutil.which("convert_hf_to_gguf.py")
        if which_result:
            which_path = Path(which_result)
            searched_paths.append(which_path)
            if which_path.exists():
                convert_script = which_path

    if convert_script is None:
        # Final fallback to the two well-known paths.
        for path in [
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path("/usr/local/bin/convert_hf_to_gguf.py"),
        ]:
            searched_paths.append(path)
            if path.exists():
                convert_script = path
                break

    if convert_script:
        logger.warning(f"Using llama.cpp convert script: {convert_script}")
        # Run conversion
        cmd = [
            "python",
            str(convert_script),
            str(merged_path),
            "--outfile",
            str(gguf_path),
            "--outtype",
            quant_str,
        ]
        try:
            # BRIDGE-A-004: Popen-based runner so Ctrl+C during a 30-min
            # quantization actually stops the child instead of leaving a
            # zombie that holds VRAM + disk for the rest of the timeout.
            result = _run_subprocess_interruptible(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            # Clean up on timeout
            shutil.rmtree(merged_path, ignore_errors=True)
            raise GGUFExportError(
                "llama.cpp conversion timed out after 30 minutes",
                output_path=str(output_path),
                quantization=quant_str,
                suggestion="The model may be too large for conversion, or llama.cpp may be stuck. Try a smaller model or check system resources."
            )
        except subprocess.CalledProcessError as e:
            # Clean up on failure
            shutil.rmtree(merged_path, ignore_errors=True)
            error_output = e.stderr[:500] if e.stderr else "No error output"
            raise GGUFExportError(
                f"llama.cpp conversion failed (exit code {e.returncode}):\n{error_output}",
                output_path=str(output_path),
                quantization=quant_str,
                suggestion="Check that llama.cpp is properly installed and up to date"
            ) from e
    else:
        # Clean up and raise
        shutil.rmtree(merged_path, ignore_errors=True)
        # BRIDGE-B-014 (Stage C): enumerate every path the discovery probed
        # so the operator knows what to populate. The
        # BACKPROPAGATE_LLAMA_CPP_PATH escape hatch is named explicitly in
        # the suggestion because operators on Windows / unusual install
        # locations need a way to point at their existing install without
        # symlinking into ~/llama.cpp.
        searched_summary = (
            "; ".join(str(p) for p in searched_paths)
            if searched_paths
            else "(no paths probed — env var unset, shutil.which returned None)"
        )
        raise GGUFExportError(
            "GGUF export requires either Unsloth or llama.cpp",
            output_path=str(output_path),
            suggestion=(
                "Install Unsloth (`pip install unsloth`) or clone llama.cpp. "
                f"Paths probed: {searched_summary}. "
                "To point at a non-standard install, set "
                "`BACKPROPAGATE_LLAMA_CPP_PATH=/path/to/convert_hf_to_gguf.py` "
                "(file or directory accepted). Default home-dir target: "
                "~/llama.cpp/convert_hf_to_gguf.py."
            )
        )

    # Clean up temp merged model
    try:
        shutil.rmtree(merged_path)
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory {merged_path}: {e}")

    # Validate output
    if not gguf_path.exists():
        raise GGUFExportError(
            f"GGUF file was not created: {gguf_path}",
            output_path=str(output_path),
            quantization=quant_str,
        )

    export_time = time.time() - start_time
    size_mb = _get_dir_size_mb(gguf_path)

    logger.info(f"GGUF exported via llama.cpp to {gguf_path} ({size_mb:.1f} MB)")

    # F-004: model card next to the GGUF (llama.cpp fallback path).
    _maybe_write_model_card(
        gguf_path,
        enabled=emit_model_card,
        run_id=run_id,
        base_model=base_model,
        output_root=output_root,
        extra_card_fields={
            "export_format": "gguf",
            "quantization": quant_str,
        },
    )

    return ExportResult(
        format=ExportFormat.GGUF,
        path=gguf_path,
        size_mb=size_mb,
        quantization=quant_str,
        export_time_seconds=export_time,
    )


def create_modelfile(
    gguf_path: str | Path,
    output_path: str | Path | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    context_length: int = 4096,
) -> Path:
    """
    Create Ollama Modelfile for the GGUF.

    Args:
        gguf_path: Path to the GGUF file
        output_path: Path to write Modelfile (default: same dir as GGUF)
        system_prompt: Optional system prompt
        temperature: Model temperature (default: 0.7)
        context_length: Context window size (default: 4096)

    Returns:
        Path to the created Modelfile
    """
    gguf_path = Path(gguf_path).resolve()

    if output_path:
        modelfile_path = Path(output_path)
    else:
        modelfile_path = gguf_path.parent / "Modelfile"

    # BRIDGE-A-001: escape backslash and double-quote in the FROM path so a
    # gguf_path containing either character (rare on POSIX, common on Windows
    # via UNC paths) produces a well-formed Modelfile. Order matters: escape
    # backslashes FIRST, otherwise the inserted escape-backslashes are
    # themselves doubled by the quote escape.
    gguf_path_str = str(gguf_path).replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        f'FROM "{gguf_path_str}"',
        "",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER num_ctx {context_length}",
    ]

    if system_prompt:
        # Escape backslashes then quotes in system prompt for the same reason.
        escaped_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
        lines.extend(
            [
                "",
                f'SYSTEM "{escaped_prompt}"',
            ]
        )

    modelfile_path.write_text("\n".join(lines))
    return modelfile_path


def register_with_ollama(
    gguf_path: str | Path,
    model_name: str,
    system_prompt: str | None = None,
) -> bool:
    """
    Register GGUF with Ollama.

    Creates Modelfile and runs `ollama create`.

    Args:
        gguf_path: Path to the GGUF file
        model_name: Name for the Ollama model
        system_prompt: Optional system prompt

    Returns:
        True if successful

    Raises:
        OllamaRegistrationError: If registration fails
    """
    # BRIDGE-A-001: refuse option-injection / control-char model names BEFORE
    # we hand the argv to the ollama CLI. ExportError (with INPUT_VALIDATION_FAILED
    # code) is the right surface — the operator gets a structured error instead
    # of a confusing "ollama: unknown option -h" or worse.
    _validate_model_name(model_name)

    gguf_path = Path(gguf_path).resolve()

    if not gguf_path.exists():
        raise OllamaRegistrationError(
            model_name,
            f"GGUF file not found: {gguf_path}",
            suggestion="Check that the GGUF export completed successfully"
        )

    # Check if Ollama is available
    if not shutil.which("ollama"):
        raise OllamaRegistrationError(
            model_name,
            "Ollama CLI not found in PATH",
            suggestion="Install Ollama from https://ollama.ai and ensure it's in your PATH"
        )

    # BRIDGE-A-003: initialize before the inner try so the finally clause can
    # safely test it. Previously, if create_modelfile() raised, the finally
    # would reference an unbound local and the user saw an UnboundLocalError
    # instead of the real OllamaRegistrationError.
    modelfile_path: Path | None = None

    # Create Modelfile
    try:
        modelfile_path = create_modelfile(gguf_path, system_prompt=system_prompt)
    except Exception as e:
        raise OllamaRegistrationError(
            model_name,
            f"Failed to create Modelfile: {e}",
        ) from e

    try:
        # BRIDGE-A-004: Popen-based runner so Ctrl+C actually propagates to
        # the ollama child instead of leaving a zombie quantization process.
        result = _run_subprocess_interruptible(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        logger.info(f"Successfully registered model '{model_name}' with Ollama")
        return True
    except subprocess.TimeoutExpired:
        raise OllamaRegistrationError(
            model_name,
            "ollama create timed out after 10 minutes",
            suggestion="The model may be too large or Ollama may be unresponsive. Check 'ollama serve' status and try again."
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr[:500] if e.stderr else "Unknown error"
        raise OllamaRegistrationError(
            model_name,
            f"ollama create failed: {error_msg}",
            suggestion="Ensure Ollama is running (ollama serve) and try again"
        ) from e
    finally:
        # BRIDGE-A-003: guard against the modelfile_path = None path (which
        # only triggers when create_modelfile() raised above — but the `raise`
        # in the except branch already short-circuits before we reach here.
        # We still guard defensively because future refactors may move the
        # raise inside a wider try.). Also tolerate transient OSErrors during
        # cleanup so they don't mask the real OllamaRegistrationError.
        if modelfile_path is not None:
            try:
                if modelfile_path.exists():
                    modelfile_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to clean up Modelfile at {modelfile_path}: {e}")


def list_ollama_models() -> list[str]:
    """
    List models registered with Ollama.

    Returns:
        List of model names
    """
    if not shutil.which("ollama"):
        return []

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

        # Parse output (skip header line)
        models = []
        lines = result.stdout.strip().split("\n")
        for line in lines[1:]:  # Skip header
            if line.strip():
                # First column is model name
                parts = line.split()
                if parts:
                    models.append(parts[0])
        return models
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
