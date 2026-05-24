# Coverage Test Roadmap

**Status**: Living document; updated 2026-05-23 (v1.3 Wave 1 Stage A).

**Current floor**: `fail_under = 50` in coverage config. Per-module gaps below
are advisory; the floor is what CI enforces.

**v1.1.0 → v1.2.0 migration note** — the legacy `ui.py` + `theme.py` Gradio
modules were preserved as `ui_gradio_legacy.py` through v1.1.x and DELETED in
v1.2.0 (Wave 4.5). Test coverage for the Reflex Web UI lives in
`backpropagate/ui_app/` and is exercised by `tests/test_ui_security.py`,
`tests/test_auth_middleware.py`, `tests/test_runs_command.py`, and the
Reflex-state tests inside `test_ui_app_*` files as they land. Any roadmap
section referencing `ui.py` / `gradio.Tab` / `TestGPUMonitoringUI` (as
Gradio components) has been removed — those surfaces no longer exist.

**Priority surfaces (post-v1.2.0)**:

| Module | Notes |
|--------|-------|
| `cli.py` | 2900+ lines; ~30 subcommands; high-traffic. Wave 1 added `--error-codes` + `runs` + `--host` gate coverage. |
| `datasets.py` | JSONL/ShareGPT/Alpaca/OpenAI auto-detect; format-drift surface. |
| `config.py` | Pydantic-settings; presets; env-var override paths. |
| `multi_run.py` | SLAO + replay strategies; resume_from + run_history coordination. |
| `ui_app/` (Reflex) | Replaces deleted `ui.py`. Covered by `test_ui_security.py`, `test_auth_middleware.py`, `test_runs_command.py`. |
| `ui_state.py` | Reflex state classes (`AppState`/`TrainState`/`MultiRunState`/`ExportState`/`DatasetState`/`RunsState`). |
| `ui_app/auth.py` | ASGI middleware — Host/Origin allowlist, cookie/HMAC session, pre-accept WS close. |

---

## 1. CLI.PY Coverage Areas

### 1.1 Color Support Detection (`_supports_color`)

```python
class TestSupportsColorExtended:
    """Extended tests for terminal color detection."""

    def test_windows_terminal_detected(self):
        """WT_SESSION env var enables colors on Windows."""

    def test_no_color_env_disables_colors(self):
        """NO_COLOR=1 should disable colors."""

    def test_force_color_env_enables_colors(self):
        """FORCE_COLOR=1 should enable colors."""

    def test_non_tty_stream_no_colors(self):
        """Piped output should not use colors."""

    def test_missing_isatty_attribute(self):
        """Handle streams without isatty method."""
```

### 1.2 Training Command Error Handling

```python
class TestCmdTrainErrors:
    """Error handling in cmd_train."""

    def test_dataset_error_shows_suggestion(self):
        """DatasetError with suggestion field displays it."""

    def test_training_error_shows_message(self):
        """TrainingError displays error message cleanly."""

    def test_backpropagate_error_generic(self):
        """BackpropagateError base class handling."""

    def test_verbose_flag_shows_traceback(self):
        """--verbose flag includes full stack trace."""

    def test_keyboard_interrupt_graceful(self):
        """Ctrl+C during training exits gracefully."""

    def test_exception_chaining_preserved(self):
        """Original exception cause preserved in output."""
```

### 1.3 Multi-Run Command Errors

```python
class TestCmdMultiRunErrors:
    """Error handling in cmd_multi_run."""

    def test_config_validation_failure(self):
        """Invalid config values raise ValidationError."""

    def test_abort_mid_run(self):
        """Aborting during a run cleans up properly."""

    def test_on_run_complete_callback_error(self):
        """Callback errors don't crash the command."""

    def test_missing_data_path(self):
        """Clear error when data path doesn't exist."""
```

### 1.4 Export Command

```python
class TestCmdExportExtended:
    """Extended export command tests."""

    def test_path_traversal_blocked(self):
        """Paths with ../ are rejected."""

    def test_invalid_format_rejected(self):
        """Unknown export formats raise error."""

    def test_ollama_registration_failure(self):
        """Ollama registration error handled gracefully."""

    def test_export_mid_process_failure(self):
        """Partial export cleanup on failure."""

    def test_invalid_quantization_param(self):
        """Invalid quantization type rejected."""
```

### 1.5 Info Command

Wave 1 (v1.3) added `--error-codes` regression coverage to
`tests/test_cli.py::TestCmdInfoErrorCodes`. Outstanding:

```python
class TestCmdInfoExtended:
    """Extended info command tests."""

    def test_no_gpu_shows_message(self):
        """Shows 'No GPU available' when CUDA unavailable."""

    def test_pynvml_missing_graceful(self):
        """Missing pynvml shows N/A for temperature."""

    def test_feature_flags_displayed(self):
        """All feature flags shown with status."""

    def test_env_vars_flag_lists_every_var(self):
        """--env-vars enumerates every BACKPROPAGATE_* env var."""

    def test_json_flag_produces_parseable_payload(self):
        """--json output round-trips through json.loads()."""
```

### 1.6 UI Command (Reflex subprocess launcher)

`cmd_ui` shells out to `python -m reflex` after validating the auth /
host / share gates. v1.3 Wave 1 added `tests/test_host_gate.py` covering
the `--host <non-loopback>` refuse-to-start path; the `--share`
refuse-to-start gate has long-standing coverage in
`tests/test_cli_extended.py::TestCmdUI::test_cmd_ui_share_without_auth_still_refuses`
(and its alias `test_cmd_ui_share_refuses_to_start`).

```python
class TestCmdUI:
    """UI launch command tests."""

    def test_auth_invalid_format(self):
        """--auth string without colon raises EXIT_USER_ERROR."""

    def test_reflex_import_error(self):
        """Missing reflex extra shows actionable error."""

    def test_subprocess_launch_failure_handled(self):
        """Reflex subprocess crash surfaces a clean error."""

    def test_keyboard_interrupt_during_ui(self):
        """Ctrl+C during UI runtime exits cleanly."""

    def test_env_vars_propagate_to_subprocess(self):
        """BACKPROPAGATE_UI_* env vars propagate correctly."""
```

### 1.7 Windows-Specific Config

```python
class TestWindowsConfigDisplay:
    """Windows-specific configuration display."""

    def test_windows_settings_shown_on_nt(self):
        """Windows settings displayed on Windows platform."""

    def test_windows_settings_hidden_on_posix(self):
        """Windows settings not shown on Linux/Mac."""
```

### 1.8 `backprop runs` / `_build_runs_payload` (added v1.3 Wave 1)

Coverage in `tests/test_runs_command.py` — pins the BRIDGE-F-001 versioned
JSON payload contract. Outstanding work:

- `cmd_show_run` (F-003) — by-id lookup + `--json` payload (not yet
  covered).
- `cmd_runs` `--limit` cap behaviour (currently only filter + sort tests).
- Pretty-print human-view branch (currently only `--json` branch tested).

---

## 2. DATASETS.PY Coverage Areas

(Original 2.1 → 2.9 sections retained — these are still load-bearing for
the dataset surface. The format-detection / dedup / curriculum / replay
families are stable post-v1.2.)

### 2.1 Validation Summary

```python
class TestValidationSummary:
    """Validation result rendering."""

    def test_empty_error_list(self):
        """Empty errors renders cleanly."""

    def test_error_truncation_at_five(self):
        """More than 5 errors shows 'and N more'."""

    def test_warning_vs_error_classification(self):
        """Warnings and errors separated correctly."""
```

### 2.2 Quality Filtering

```python
class TestQualityFiltering:
    """Dataset quality filtering."""

    def test_filter_by_min_tokens(self):
        """Samples below min_tokens filtered out."""

    def test_filter_by_max_tokens(self):
        """Samples above max_tokens filtered out."""

    def test_filter_by_min_turns(self):
        """Samples with few turns filtered out."""

    def test_filter_missing_assistant(self):
        """Samples without assistant response filtered."""

    def test_custom_filter_callback(self):
        """Custom filter function applied correctly."""

    def test_combined_filters(self):
        """Multiple filters applied together."""

    def test_empty_sample_list(self):
        """Empty input returns empty output."""

    def test_extreme_token_counts(self):
        """Very short and very long texts handled."""
```

### 2.3 Format Detection

```python
class TestFormatDetection:
    """Dataset format auto-detection."""

    def test_detect_sharegpt_format(self):
        """ShareGPT format detected from structure."""

    def test_detect_alpaca_format(self):
        """Alpaca format detected from keys."""

    def test_detect_openai_format(self):
        """OpenAI messages format detected."""

    def test_detect_chatml_format(self):
        """ChatML format detected from tags."""

    def test_detect_raw_text_format(self):
        """Plain text format as fallback."""

    def test_mixed_format_error(self):
        """Mixed formats raise ValidationError."""

    def test_corrupted_json_handling(self):
        """Malformed JSON handled gracefully."""

    def test_file_permission_error(self):
        """Permission denied handled gracefully."""

    def test_encoding_error_handling(self):
        """Non-UTF8 files handled with fallback."""
```

### 2.4 Deduplication

```python
class TestDeduplication:
    """Exact and fuzzy deduplication."""

    def test_exact_dedup_removes_duplicates(self):
        """Identical samples removed."""

    def test_minhash_dedup_similar_samples(self):
        """Similar samples detected and removed."""

    def test_minhash_threshold_boundary(self):
        """Threshold 0.0 and 1.0 edge cases."""

    def test_datasketch_not_installed(self):
        """Graceful error when datasketch missing."""

    def test_empty_text_handling(self):
        """Empty strings don't crash dedup."""

    def test_ngram_generation_short_text(self):
        """Very short texts generate valid n-grams."""
```

### 2.5 Perplexity Filtering

```python
class TestPerplexityFiltering:
    """Perplexity-based quality filtering."""

    def test_model_loading_success(self):
        """GPT-2 model loads correctly."""

    def test_model_loading_failure(self):
        """Missing model raises helpful error."""

    def test_score_computation_normal(self):
        """Perplexity scores computed correctly."""

    def test_score_computation_short_text(self):
        """Very short texts handled gracefully."""

    def test_batch_processing(self):
        """Batch scoring works correctly."""

    def test_percentile_threshold(self):
        """Percentile-based filtering works."""

    def test_absolute_threshold(self):
        """Absolute perplexity threshold works."""

    def test_inf_score_handling(self):
        """Infinite scores handled gracefully."""

    def test_no_valid_scores(self):
        """All failed scores handled."""

    def test_lazy_model_loading(self):
        """Model only loaded when needed."""

    def test_device_autodetect(self):
        """CUDA vs CPU automatically selected."""
```

### 2.6 File Loading

```python
class TestDatasetLoading:
    """Dataset file loading."""

    def test_load_jsonl_file(self):
        """JSONL file loaded correctly."""

    def test_load_json_array(self):
        """JSON array file loaded correctly."""

    def test_load_json_single_object(self):
        """Single JSON object handled."""

    def test_load_parquet_file(self):
        """Parquet file loaded with pandas."""

    def test_load_csv_file(self):
        """CSV file loaded with pandas."""

    def test_load_text_file(self):
        """Plain text file split by double newlines."""

    def test_pandas_not_installed(self):
        """Helpful error when pandas missing."""

    def test_large_file_sampling(self):
        """Large files sampled for format detection."""

    def test_utf8_encoding_error(self):
        """Non-UTF8 files show encoding error."""
```

### 2.7 Streaming Datasets

```python
class TestStreamingDatasets:
    """HuggingFace streaming datasets."""

    def test_stream_from_hub(self):
        """Stream dataset from HuggingFace Hub."""

    def test_stream_jsonl_file(self):
        """Stream from local JSONL file."""

    def test_streaming_failure(self):
        """Network error handled gracefully."""

    def test_iterator_exhaustion(self):
        """Exhausted iterator handled."""

    def test_take_and_skip(self):
        """take() and skip() work correctly."""

    def test_empty_batch(self):
        """Empty batch handled gracefully."""
```

### 2.8 Curriculum Learning

```python
class TestCurriculumLearning:
    """Curriculum-based data ordering."""

    def test_difficulty_scoring(self):
        """Difficulty scores computed correctly."""

    def test_empty_text_difficulty(self):
        """Empty text returns default score."""

    def test_single_word_difficulty(self):
        """Single word text handled."""

    def test_vocabulary_complexity(self):
        """Vocabulary complexity calculated."""

    def test_chunk_creation(self):
        """Data split into difficulty chunks."""

    def test_last_chunk_remainder(self):
        """Last chunk with fewer samples handled."""

    def test_analysis_without_reorder(self):
        """analyze_only mode doesn't reorder."""
```

### 2.9 Statistics

```python
class TestDatasetStatistics:
    """Dataset statistics computation."""

    def test_empty_dataset_stats(self):
        """Empty dataset returns zero stats."""

    def test_token_counting(self):
        """Token counts calculated correctly."""

    def test_turn_counting(self):
        """Conversation turns counted."""

    def test_system_prompt_extraction(self):
        """System prompts identified."""

    def test_unique_system_prompts(self):
        """Unique system prompt set computed."""
```

---

## 3. CONFIG.PY Coverage Areas

### 3.1 Pydantic Availability

```python
class TestPydanticFallback:
    """Config loading without pydantic."""

    def test_pydantic_available(self):
        """Uses pydantic-settings when available."""

    def test_pydantic_unavailable_fallback(self):
        """Falls back to dataclass config."""

    def test_env_var_parsing_fallback(self):
        """Environment variables parsed manually."""
```

### 3.2 Fallback Configuration

```python
class TestFallbackConfig:
    """Dataclass-based config fallback."""

    def test_get_env_str(self):
        """String env vars parsed correctly."""

    def test_get_env_int(self):
        """Integer env vars parsed correctly."""

    def test_get_env_float(self):
        """Float env vars parsed correctly."""

    def test_get_env_bool(self):
        """Boolean env vars parsed correctly."""

    def test_invalid_int_value(self):
        """Non-numeric string for int raises error."""

    def test_default_fallback(self):
        """Missing env var uses default value."""
```

### 3.3 Training Arguments

```python
class TestTrainingArgs:
    """Training argument generation."""

    def test_get_training_args_basic(self):
        """Basic training args generated."""

    def test_windows_dataloader_workers(self):
        """Windows uses 0 dataloader workers."""

    def test_linux_dataloader_workers(self):
        """Linux can use multiple workers."""
```

### 3.4 Settings Management

```python
class TestSettingsManagement:
    """Settings reload and caching."""

    def test_reload_settings(self):
        """reload_settings() clears cache."""

    def test_env_var_override(self):
        """Environment variables override defaults."""

    def test_settings_singleton(self):
        """Settings returns same instance."""
```

### 3.5 Presets

```python
class TestTrainingPresets:
    """Training preset configurations."""

    def test_fast_preset(self):
        """Fast preset has low steps, high lr."""

    def test_balanced_preset(self):
        """Balanced preset has medium values."""

    def test_quality_preset(self):
        """Quality preset has high steps, low lr."""
```

---

## 4. MULTI_RUN.PY Coverage Areas

### 4.1 Initialization

```python
class TestMultiRunInit:
    """MultiRunTrainer initialization."""

    def test_checkpoint_manager_init(self):
        """Checkpoint manager created correctly."""

    def test_slao_merger_init(self):
        """SLAO merger initialized when enabled."""

    def test_gpu_monitor_start(self):
        """GPU monitor starts with training."""

    def test_preflight_gpu_check(self):
        """GPU status checked before training."""
```

### 4.2 Dataset Loading

```python
class TestMultiRunDataset:
    """Dataset loading for multi-run."""

    def test_dataset_loading_success(self):
        """Dataset loads from path."""

    def test_dataset_loading_failure(self):
        """Missing dataset raises DatasetError."""

    def test_dataset_validation(self):
        """Dataset validated before training."""
```

### 4.3 Run Execution

```python
class TestRunExecution:
    """Individual run execution."""

    def test_data_chunk_creation(self):
        """Data chunk created for run."""

    def test_replay_samples_added(self):
        """Experience replay samples included."""

    def test_learning_rate_linear_decay(self):
        """Linear LR decay calculated correctly."""

    def test_learning_rate_cosine_decay(self):
        """Cosine LR decay calculated correctly."""

    def test_lora_state_operations(self):
        """LoRA state dict saved and loaded."""

    def test_sft_trainer_creation(self):
        """SFTTrainer created with correct args."""

    def test_training_execution(self):
        """Training runs without error."""

    def test_loss_history_aggregation(self):
        """Loss values collected across steps."""

    def test_checkpoint_saving(self):
        """Checkpoint saved after run."""
```

### 4.4 Run Management

```python
class TestRunManagement:
    """Multi-run management."""

    def test_validation_loss_computation(self):
        """Validation loss computed correctly."""

    def test_early_stopping_triggered(self):
        """Training stops when loss plateaus."""

    def test_cooldown_between_runs(self):
        """Cooldown period enforced."""

    def test_gpu_callback_invocation(self):
        """GPU status callbacks fired."""

    def test_run_result_creation(self):
        """RunResult created with all fields."""
```

### 4.5 Experience Replay

```python
class TestExperienceReplay:
    """Experience replay strategies."""

    def test_replay_recent_strategy(self):
        """Recent samples selected."""

    def test_replay_random_strategy(self):
        """Random samples selected."""

    def test_replay_all_previous_strategy(self):
        """All previous samples included."""

    def test_replay_sample_count(self):
        """Correct number of samples selected."""

    def test_replay_empty_history(self):
        """First run has no replay samples."""
```

### 4.6 Data Pipeline

```python
class TestDataPipeline:
    """Data chunking and wrapping."""

    def test_chunk_wrapping(self):
        """Data wraps when chunk exceeds dataset."""

    def test_shuffle_behavior(self):
        """Shuffling randomizes order."""

    def test_validation_subset_selection(self):
        """Validation subset selected correctly."""

    def test_validation_loss_computation(self):
        """Validation loss computed on subset."""

    def test_device_handling(self):
        """Tensors moved to correct device."""
```

### 4.7 CLI Interface

```python
class TestMultiRunCLI:
    """Multi-run CLI interface."""

    def test_argument_parsing(self):
        """CLI arguments parsed correctly."""

    def test_config_creation(self):
        """SpeedrunConfig created from args."""

    def test_run_execution(self):
        """trainer.run() called correctly."""

    def test_result_display(self):
        """Results formatted for display."""
```

---

## 5. UI_APP/ Coverage Areas (Reflex — replaces deleted `ui.py`)

The Reflex Web UI shipped in v1.1.0 and is the canonical UI surface from
v1.2.0 onward. Coverage lives in:

- `tests/test_ui_security.py` — `EnhancedRateLimiter`, `FileValidator`,
  shared auth/path-sandbox helpers in `backpropagate.ui_security`.
- `tests/test_auth_middleware.py` — ASGI middleware contract per
  `DESIGN_BRIEF.md` "Testing requirements" (16 brief-numbered tests +
  hardening additions; v1.3 Wave 1 added pre-/post-accept HMAC
  regression set).
- `tests/test_runs_command.py` (v1.3 Wave 1) — `RunsState` + CLI runs
  data API (BRIDGE-F-001).
- `tests/test_host_gate.py` (v1.3 Wave 1) — `--host <non-loopback>`
  refuse-to-start gate (BRIDGE-A-002).

### 5.1 Reflex State Classes (`ui_state.py`)

```python
class TestAppState:
    """Top-level AppState (theme, active surface)."""

    def test_initial_theme_default(self):
        """Theme defaults to the documented value."""

    def test_set_active_surface_updates_state(self):
        """Switching surfaces (train/runs/export) updates state."""


class TestTrainState:
    """TrainState — single-run training surface."""

    def test_form_field_defaults(self):
        """Empty initial state — no model/data/etc."""

    def test_set_model_validates_shape(self):
        """Invalid model strings are rejected."""

    def test_start_training_dispatches_subprocess(self):
        """start_training() shells out to backprop train ..."""


class TestMultiRunState:
    """MultiRunState — SLAO multi-run surface."""

    def test_runs_default_5(self):
        """Default number of runs is the documented value."""

    def test_replay_strategy_dropdown_options(self):
        """Replay strategy options match the SLAO contract."""


class TestExportState:
    """ExportState — LoRA/merged/GGUF export surface."""

    def test_format_options(self):
        """Export format dropdown lists all supported formats."""

    def test_quantization_visible_for_gguf(self):
        """Quant dropdown shows only when format=gguf."""


class TestDatasetState:
    """DatasetState — dataset upload/preview surface."""

    def test_uploaded_path_validation(self):
        """Operator-supplied paths are sandbox-validated."""

    def test_detected_format_after_upload(self):
        """Format detection populates detected_format."""


class TestRunsState:  # Covered by tests/test_runs_command.py — see there.
    """Run-history surface — populates the /runs page."""
```

### 5.2 ASGI Auth Middleware (`ui_app/auth.py`)

See `tests/test_auth_middleware.py` for the 16 brief-numbered tests +
hardening additions (cookie tampering, expired-cookie pre-accept close,
HMAC compare_digest source-level invariant — added v1.3 Wave 1
TESTS-A-005 / A-006).

### 5.3 Reflex Subprocess Launcher (`cli.cmd_ui`)

See `tests/test_cli_extended.py::TestCmdUI::test_cmd_ui_share_without_auth_still_refuses`
(—share + auth gates) and `tests/test_host_gate.py` (—host gate, v1.3
Wave 1).

---

## Implementation Priority

### Phase 1: Critical Path (Wave 1 — partially shipped)
- [x] CLI `--error-codes` regression (TESTS-A-003)
- [x] CLI `--host` gate coverage (BRIDGE-A-002)
- [x] CLI `runs` + `_build_runs_payload` (TESTS-A-002)
- [x] Auth middleware pre-/post-accept HMAC regression (TESTS-A-005/A-006)
- [x] `test_e2e_resume_from_checkpoint` rewrite (TESTS-A-004)
- [ ] Dataset format-detection drift coverage (next wave)
- [ ] Multi-run execution edge cases (next wave)

### Phase 2: Feature Coverage
- Dataset quality filtering / dedup / replay edge cases.
- ui_state.py Reflex state classes (TrainState/MultiRunState/ExportState).
- cmd_ui subprocess-launch failure paths.

### Phase 3: Edge Cases
- Perplexity filtering / streaming datasets / curriculum learning.
- ui_app/auth.py — cookie persistence across restarts, lock-file mode 0600
  on POSIX (already covered) + a Windows-ACL companion.

### Phase 4: Polish
- File loading edge cases, Windows-specific tests, statistics edge cases.

---

## Test Utilities

Current helpers live in `tests/helpers/` (asgi.py, callbacks.py, ws.py).
Add Reflex state helpers as the ui_state.py coverage rolls out.
