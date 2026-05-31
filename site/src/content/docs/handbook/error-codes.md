---
title: Error codes
description: Catalog of stable Backpropagate error codes — what each one means and how to fix it.
sidebar:
  order: 5
---

Every Backpropagate exception carries a stable, machine-readable `code` you can search for, grep your logs for, or quote in a bug report. Codes are grouped by prefix:

- `INPUT_*` — bad input from the user (your command line, your dataset, your config). Exit code `1`. Not retryable.
- `CONFIG_*` — invalid persisted configuration. Exit code `1`. Not retryable.
- `DEP_*` — an external dependency (HuggingFace Hub, Ollama daemon, CUDA driver) misbehaved. Exit code `2`. Often retryable.
- `RUNTIME_*` — something failed inside Backpropagate or the training/export pipeline. Exit code `2`.
- `STATE_*` — corrupt on-disk state (checkpoint file, SLAO snapshot). Exit code `2`. Not retryable without manual cleanup.
- `PARTIAL_*` — operation finished, but not every unit succeeded. Exit code `3`.

You will see codes printed in stderr as `[CODE_NAME]: message` and in the structured envelope returned by `BackpropagateError.to_dict()`. They are designed to stay stable across class renames — quote them in issues.

## INPUT_*  (user-actionable)

| Code | Raised when | Fix |
|------|-------------|-----|
| `INPUT_VALIDATION_FAILED` | A user-supplied argument or flag failed validation (e.g. `steps=0`, malformed `--auth`). | Re-read the suggestion in stderr; fix the offending argument. |
| `INPUT_AUTH_REQUIRED` | An operation required `--auth` credentials (or `BACKPROPAGATE_UI_AUTH=user:pass`) but they were not supplied. | Pass `--auth user:password` on the CLI, or set `BACKPROPAGATE_UI_AUTH=user:pass` in the environment. See [handbook/security.md](/backpropagate/handbook/security/) for the full auth contract. |
| `INPUT_AUTH_INVALID_SHAPE` | The credentials passed to the UI launcher are not a `username:password` tuple. | Use the format `--auth username:password` (single colon, no spaces). |
| `INPUT_DATASET_INVALID` | The dataset is malformed in a way that doesn't fit a more specific code. | Inspect the file; ensure each line is valid JSON for JSONL inputs. |
| `INPUT_DATASET_NOT_FOUND` | The dataset path does not exist. | Check the path is spelled correctly and the file exists. Relative paths resolve against the current working directory. |
| `INPUT_DATASET_PARSE_FAILED` | A line in a JSONL/CSV dataset failed to parse. | The error includes the line number; open the file and fix the offending row. |
| `INPUT_DATASET_VALIDATION_FAILED` | Dataset content failed schema validation (missing required fields, wrong shape). | See the listed validation errors in the suggestion and fix them. |
| `INPUT_DATASET_FORMAT_UNSUPPORTED` | Dataset format could not be auto-detected as ShareGPT / Alpaca / OpenAI chat / ChatML / raw text. | Convert to one of the supported formats, or pass an already-loaded HuggingFace `Dataset` object. |
| `INPUT_DATASET_REPORT_THRESHOLD` | A `backprop data report` quality gate tripped — a `--fail-on-dups` / `--fail-on-contamination` / `--max-outlier-rate` threshold was exceeded, or `--strict` promoted a WARN verdict to FAIL. Returned as exit `65` (not raised); the code is stamped into the structured log line. | Inspect the report's `failed_thresholds` list (re-run without `--json` for the human summary). Either clean the dataset (dedupe / trim outliers / remove contamination against the `--against` set) or relax the threshold you passed. Drop the `--fail-*` flags to run in advisory mode (exit `0`). |
| `INPUT_EVAL_RUN_NOT_FOUND` | `backprop eval <run_id>` (or `--vs` / `--gate-against`) named a run_id that is not present in the on-disk run history under the configured `--output` directory. | Run `backprop runs --output <dir>` to list available run_ids. If the run was trained under a different output directory, re-run with `--output <that-dir>`. Partial-prefix matches are accepted as long as the prefix is unambiguous. |
| `INPUT_EVAL_HELDOUT_UNRESOLVED` | `backprop eval` could not resolve the held-out evaluation set — the `--heldout` path is missing / unreadable, or the `--prompts` file could not be opened. | Pass a readable path to `--heldout` (a JSONL held-out split) or `--prompts` (one prompt per line, or JSONL). Check the path resolves from the current working directory and is UTF-8. |

## CONFIG_*  (configuration)

| Code | Raised when | Fix |
|------|-------------|-----|
| `CONFIG_INVALID` | The persisted configuration (env vars, `.env`, settings) is invalid in a way that doesn't fit a more specific code. | Run `backprop config` to dump the resolved config; correct the offending value. |
| `CONFIG_INVALID_SETTING` | A specific setting has an invalid value (type, range, enum). | The error includes the setting name, the value seen, and the expected shape — fix that one knob. |

## DEP_*  (external dependency)

| Code | Raised when | Fix |
|------|-------------|-----|
| `DEP_MODEL_LOAD_FAILED` | The model could not be loaded from HuggingFace Hub or local cache. Common causes: gated model without `HF_TOKEN`, network timeout, typo in model name, corrupted cache. | Verify the model name. For gated models (e.g. some Llama variants) run `huggingface-cli login` and re-try. Retryable — the trainer will back off on 5xx/429. |
| `DEP_DATASET_ENGINE_MISSING` | An optional dependency needed to read a dataset file format (pandas for CSV; pandas + a pyarrow parquet engine for parquet) is not installed. | Install the missing extra: `pip install pandas pyarrow` (parquet) or `pip install pandas` (CSV). |
| `DEP_GPU_NOT_AVAILABLE` | CUDA / a compatible GPU could not be detected. | Confirm `nvidia-smi` works. Reinstall PyTorch with the CUDA wheel that matches your driver (`pip install torch --index-url https://download.pytorch.org/whl/cu121` etc.). |
| `DEP_OLLAMA_REGISTRATION_FAILED` | `register_with_ollama(...)` could not reach the Ollama daemon or the registration call returned an error. | Start the daemon: `ollama serve` (or install Ollama from <https://ollama.com>). Default endpoint is `localhost:11434`. Retryable. |
| `DEP_MLX_UNAVAILABLE` | **v1.5 T3.1 (experimental)** — the MLX (Apple-Silicon) rail was selected (resolved `backend='mlx'`) but the `mlx_lm` toolchain is not importable on this host. `mlx-lm` is **Apple-Silicon-ONLY** (macOS + arm64), so on a Windows / Linux / Intel-Mac host the `[mlx]` extra cannot install and this fires. Raised by `MLXBackend.run()` at the subprocess-launch attempt; a *forced* `backend='mlx'` on a non-Apple host is normally intercepted earlier at `Trainer` construction as `CONFIG_INVALID_SETTING`, so reaching this code usually means a corrupted `mlx-lm` install on a real Mac. | On an Apple-Silicon Mac: install the extra — `pip install 'backpropagate[mlx]'`. On a non-Apple host: use `backend='auto'` (the default — routes to CUDA) or `backend='cuda'`; `backend='mlx'` cannot run there. Set the rail via `--backend` (CLI) or `BACKPROPAGATE_TRAINING__BACKEND` (env). |

## RUNTIME_*  (internal runtime)

| Code | Raised when | Fix |
|------|-------------|-----|
| `RUNTIME_TRAINING_FAILED` | Training failed in a way that doesn't fit a more specific code. | Re-run with `--verbose` (or `BACKPROPAGATE_DEBUG=1`) to see the unredacted traceback. The `run_id` printed at startup correlates this failure with every log line and checkpoint — quote it in any bug report. |
| `RUNTIME_TRAINING_ABORTED` | Training was aborted (user interrupt, GPU pause/abort, etc.). The error includes `steps_completed` and the last `checkpoint_path` so you can resume. | If the abort was from GPU temperature or VRAM pressure, fix the underlying issue and resume from the checkpoint with `backprop resume <run_id>` (the run_id is printed in the error). |
| `RUNTIME_EVAL_GATE_REGRESSED` | `backprop eval --gate-against <baseline_run_id>` determined the evaluated run regressed the held-out metric beyond the allowed `--max-regression`. This is the eval-gate that protects continual-merge / SLAO campaigns. Returned as exit `65` (not raised); the code is stamped into the structured log line. | The after-run is worse than the baseline by more than `--max-regression`. Inspect the diff (re-run with `--vs <baseline>` for the side-by-side). Either keep the baseline (reject this run / merge), raise `--max-regression` if a small regression is acceptable, or retrain with a higher learning rate / more steps / cleaner data. |
| `RUNTIME_EVAL_FAILED` | `backprop eval` failed to complete the evaluation — the model could not be loaded, the held-out forward pass crashed, or sample generation raised. Distinct from a clean regression (`RUNTIME_EVAL_GATE_REGRESSED`); here the eval did not finish. Surfaced via the catch-all exit-code mapper (exit `2`, or `137` on CUDA OOM / `69` on a Hub failure). | Re-run with `--verbose` (or `BACKPROPAGATE_DEBUG=1`) for the full traceback. Confirm the run's checkpoint loads via `backprop show-run <run_id>` and that the model fits in VRAM at eval time (lower `--num-samples` / `--max-new-tokens` if you OOM during generation). |
| `RUNTIME_UI_AUTH_NOT_ENFORCED` | `backprop ui --share` (or `--host <non-loopback>`) was invoked without `--auth user:password`. The v1.2.0+ contract refuses to start the UI rather than expose an unauthenticated tunnel. (The pre-v1.2.0 `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` opt-out is a **no-op** under the Reflex UI — held only for forward-compat with the Gradio era — and will not relax this gate.) | Pass `--auth user:password` (or `--auth-file <path>` in v1.3+ to keep the credential out of shell history). If you don't actually need a public URL, use SSH port-forwarding instead: `ssh -L 7860:localhost:7860 <host>`. See [handbook/security.md](/backpropagate/handbook/security/) for the full threat model and the four supported auth modes. |
| `RUNTIME_EXPORT_FAILED` | Export failed in a way that doesn't fit a more specific code. | Re-run with `--verbose`; verify the model loaded cleanly first via `backprop info`. If the export step fired before `trainer.train()` returned, check the LoRA-export-without-training warning above your stderr — `save()` no longer silently writes untrained weights as of v1.4. |
| `RUNTIME_LORA_EXPORT_FAILED` | LoRA adapter export failed. | Confirm the trainer actually has LoRA adapters attached — `trainer.train()` must have run before `trainer.export(...)` or `trainer.save(...)`. As of v1.4, `save()` emits a WARN line if you call it before `train()`, naming the path it would have written; check your stderr for that line. |
| `RUNTIME_GGUF_EXPORT_FAILED` | GGUF export failed. | Install the export extra: `pip install backpropagate[export]`. On first run, `llama-cpp-python` may need to build from source — Windows needs Visual C++ Build Tools + CMake; Linux needs `build-essential` + `cmake`. See [troubleshooting → llama-cpp-python build failed](/backpropagate/handbook/troubleshooting/#llama-cpp-python-build-failed-on-first-gguf-export). |
| `RUNTIME_MERGE_EXPORT_FAILED` | Merging the LoRA back into the base model failed. | Most commonly VRAM pressure during the merge — try a smaller base model, free GPU memory first (`nvidia-smi` to see what's holding it), or pass `device_map="cpu"` to merge on CPU at the cost of speed. |
| `RUNTIME_GPU_ERROR` | Generic GPU error that doesn't fit a more specific GPU code. | Run `backprop info` to confirm CUDA / GPU / driver are wired up correctly. Then re-run training with `--verbose` to see which op tripped. If you see a CUBLAS error, see the [CUDA troubleshooting page → CUBLAS errors](/backpropagate/handbook/troubleshooting-cuda/#cublas-errors). |
| `RUNTIME_GPU_OOM` | Out-of-memory on the GPU during training, eval, or export. | The B-002 OOM-recovery path (enabled by default) auto-shrinks batch size up to 3 times before giving up. If you still see this code: lower `--batch-size` manually, reduce the model's `max_seq_length` setting (most VRAM lives in attention — pass `Trainer(max_seq_length=1024)` or set `BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH=1024`), or pick a smaller model preset (Qwen 2.5 3B fits in ~8GB). To opt out of auto-recovery and let OOM bubble up: `Trainer(oom_recovery=False)`. |
| `RUNTIME_FULL_FT_MODEL_TOO_LARGE` | `Trainer(mode="full")` was requested for a model whose parameter count exceeds the documented 3B ceiling for consumer 16GB GPUs. Fires at `Trainer.__init__` (preset-table lookup) AND a second time at `Trainer.load_model()` (authoritative `model.num_parameters()` check). | Two recoveries: (1) re-run with `mode="lora"` (the default) — LoRA fits 7B+ on a 16GB card and is the recommended path per [Biderman 2024](https://arxiv.org/abs/2405.09673) + [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/); (2) switch to a model that fits the ceiling — `phi-4-mini-3.8b` (MIT), `qwen3.5-4b` (Apache 2.0), or `smollm3-3b` (Apache 2.0). All three carry preset entries and the catalog calls each one out as commercial-safe. See [full fine-tuning](/backpropagate/handbook/full-fine-tuning/) for the quality math and the recovery decision tree. |
| `RUNTIME_FP8_UNSUPPORTED` | **v1.5 (experimental)** — the FP8 compute path (`--fp8` / `BACKPROPAGATE_TRAINING__FP8` / `Trainer(fp8=True)`) was enabled but FP8 conversion / the first float8 GEMM failed at runtime — most often a half-installed or ABI-mismatched torchao whose `import torchao.float8` fails AFTER the capability gate already promised FP8 (so it is NOT degraded silently to bf16). Other FP8 failure axes (no CUDA, sm < 9, conversion error) instead degrade to bf16 with a warning and do not raise this. | Re-run without `--fp8` (`fp8=False`) to train in bf16 — correct, just without the FP8 speed/memory win. If you want FP8: reinstall a torchao matching your torch build — `pip install --force-reinstall 'backpropagate[fp8]'`; FP8 needs base-layer inner dims divisible by 16, a Hopper/Blackwell (sm_90+) GPU, and `torch>=2.11` for the compiled torchao kernels (2.10 uses the slower `_scaled_mm` fallback). Note `--fp8` is `--mode lora` + `--method sft` only in v1.5; `mode='full'` / `method='orpo'` / explicit 4-bit combined with FP8 are rejected earlier as `CONFIG_INVALID_SETTING`. |
| `RUNTIME_GPU_TEMPERATURE_CRITICAL` | GPU exceeded the configured safety threshold and training was paused or aborted. Retryable. | Wait for the GPU to cool (training auto-resumes once it does); if it keeps tripping, improve case airflow and / or lower batch size to reduce sustained load. See [GPU safety](/backpropagate/handbook/reference/#gpu-safety) for the temperature thresholds. |
| `RUNTIME_GPU_MONITORING_FAILED` | The GPU monitor could not query NVML. | Install pynvml: `pip install pynvml`. If pynvml is installed and you still see this, your driver may not expose NVML — confirm with `nvidia-smi`. |
| `RUNTIME_SLAO_ERROR` | Generic SLAO failure. | Re-run with `--verbose` to see which merge step tripped. The `run_index` field in the error includes the run number — pair it with the on-disk checkpoint for that run when reporting. |
| `RUNTIME_SLAO_MERGE_FAILED` | SLAO weight merge failed at a specific run. | The error includes `run_index`; inspect the on-disk checkpoint for that run. |
| `SLAO_MERGE_DIVERGED` | A SLAO merge produced weights with non-finite values (NaN/Inf). Defensive check raised before the bad weights enter the model. | Reduce the learning rate, shorten `steps_per_run`, or pick a different `merge_mode`. |
| `PEFT_API_INCOMPATIBLE` | The installed `peft` version does not expose the API Backpropagate expects. | Upgrade peft: `pip install -U peft`. |
| `UI_OUTPUT_DIR_FORBIDDEN` | The `BACKPROPAGATE_UI__OUTPUT_DIR` env var (or default) resolved to a system / credential directory like `/etc`, `~/.ssh`, `C:\Windows\System32`. | Pick a non-system directory under your home (e.g. `~/.backpropagate/ui-outputs` or `~/work/backprop-out`). |

## STATE_*  (corrupt state)

| Code | Raised when | Fix |
|------|-------------|-----|
| `STATE_CHECKPOINT_INVALID` | A checkpoint file could not be saved or loaded — missing manifest, corrupt safetensors, mid-write crash. | Delete the offending checkpoint directory; if it was written via atomic-rename, look for stray `.partial` files and remove them. |
| `STATE_SLAO_CHECKPOINT_INVALID` | A SLAO checkpoint snapshot is corrupt. | Same fix — remove the bad snapshot and re-run. SLAO will rebuild from the previous good snapshot. |

## PARTIAL_*  (mixed success/failure)

| Code | Raised when | Fix |
|------|-------------|-----|
| `PARTIAL_BATCH_OPERATION` | A per-item loop (e.g. multi-run training, batch export) finished with some items failing. The error lists the first N failures and a success-rate percentage. | Inspect the listed errors; re-run the failed items individually. |
| `PARTIAL_SUCCESS` | The CLI ran to completion but not every unit succeeded. Maps to exit code `3`. | The operation completed enough to be useful — inspect logs, decide whether to retry the failed units. |

## Reading codes programmatically

Every `BackpropagateError` exposes a structured envelope:

```python
from backpropagate.exceptions import BackpropagateError

try:
    trainer.train("data.jsonl", steps=100)
except BackpropagateError as e:
    envelope = e.to_dict()
    # {
    #   "type": "GPUMemoryError",
    #   "code": "RUNTIME_GPU_OOM",
    #   "message": "Insufficient GPU memory: need 14.2GB, have 12.0GB",
    #   "retryable": False,
    #   "suggestion": "Try reducing batch size, ...",
    #   "details": {"required_gb": 14.2, "available_gb": 12.0},
    # }
```

The `retryable` flag tells callers whether a naive retry is safe. `True` for `DEP_OLLAMA_REGISTRATION_FAILED`, `RUNTIME_GPU_TEMPERATURE_CRITICAL`, `DEP_MODEL_LOAD_FAILED`; `False` for everything else by default.

## See also

- [Troubleshooting](/backpropagate/handbook/troubleshooting/) — symptoms-first reverse index (start here if you don't yet know which code fired).
- [Environment variables](/backpropagate/handbook/env-vars/) — every `BACKPROPAGATE_*` knob, including the security and UI sandbox knobs referenced above.
