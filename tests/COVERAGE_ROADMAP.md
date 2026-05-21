# Coverage Test Roadmap

**Target**: Increase coverage from 70% to 85%+

**Priority Modules**:
| Module | Current | Target | Gap |
|--------|---------|--------|-----|
| cli.py | 73% | 90% | 17% |
| datasets.py | 68% | 85% | 17% |
| config.py | 61% | 85% | 24% |
| multi_run.py | 55% | 85% | 30% |
| ui.py | 53% | 75% | 22% |

---

## 1. CLI.PY Tests (73% → 90%)

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
```python
class TestCmdInfoExtended:
    """Extended info command tests."""

    def test_no_gpu_shows_message(self):
        """Shows 'No GPU available' when CUDA unavailable."""

    def test_pynvml_missing_graceful(self):
        """Missing pynvml shows N/A for temperature."""

    def test_feature_flags_displayed(self):
        """All feature flags shown with status."""
```

### 1.6 UI Command
```python
class TestCmdUI:
    """UI launch command tests."""

    def test_auth_invalid_format(self):
        """Auth string without colon raises error."""

    def test_ui_import_error(self):
        """Missing gradio shows helpful error."""

    def test_launch_failure_handled(self):
        """Gradio launch error handled gracefully."""

    def test_keyboard_interrupt_during_ui(self):
        """Ctrl+C during UI runtime exits cleanly."""

    def test_auth_parsed_correctly(self):
        """user:pass format parsed into tuple."""
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

---

## 2. DATASETS.PY Tests (68% → 85%)

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

## 3. CONFIG.PY Tests (61% → 85%)

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

## 4. MULTI_RUN.PY Tests (55% → 85%)

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

## 5. UI.PY Tests (53% → 75%)

### 5.1 Security & Theme
```python
class TestUISecurityExtended:
    """Extended UI security tests."""

    def test_theme_creation(self):
        """Custom theme created correctly."""

    def test_css_application(self):
        """Custom CSS applied to interface."""

    def test_rate_limiting_decorator(self):
        """Rate limiter decorator works."""

    def test_input_sanitization(self):
        """User inputs sanitized."""
```

### 5.2 Training Interface
```python
class TestTrainingInterface:
    """Training tab UI components."""

    def test_model_dropdown_populated(self):
        """Model dropdown has options."""

    def test_training_params_validated(self):
        """Invalid params show error."""

    def test_start_training_button(self):
        """Start button triggers training."""

    def test_stop_training_button(self):
        """Stop button aborts training."""

    def test_progress_display_updates(self):
        """Progress bar updates during training."""
```

### 5.3 Dataset Interface
```python
class TestDatasetInterface:
    """Dataset tab UI components."""

    def test_dataset_preview(self):
        """Dataset samples displayed."""

    def test_validation_results(self):
        """Validation errors shown."""

    def test_format_detection_display(self):
        """Detected format shown."""

    def test_statistics_display(self):
        """Dataset stats displayed."""
```

### 5.4 GPU Monitoring
```python
class TestGPUMonitoringUI:
    """GPU monitoring dashboard."""

    def test_temperature_display(self):
        """Temperature shown with color."""

    def test_vram_display(self):
        """VRAM usage shown with bar."""

    def test_status_indicator(self):
        """Status indicator updates."""

    def test_history_graph(self):
        """Temperature history graphed."""
```

### 5.5 Export Interface
```python
class TestExportInterface:
    """Export tab UI components."""

    def test_format_selection(self):
        """Export format dropdown works."""

    def test_quantization_selection(self):
        """Quantization options shown for GGUF."""

    def test_export_button(self):
        """Export button triggers export."""

    def test_progress_display(self):
        """Export progress shown."""
```

### 5.6 Callbacks & Logging
```python
class TestUICallbacks:
    """UI callback system."""

    def test_training_progress_callback(self):
        """Progress callback updates UI."""

    def test_error_callback(self):
        """Error callback shows message."""

    def test_completion_callback(self):
        """Completion callback shows success."""

    def test_log_streaming(self):
        """Logs streamed to UI."""
```

---

## Implementation Priority

### Phase 1: Critical Path (Week 1)
1. CLI error handling tests
2. Dataset format detection tests
3. Multi-run execution tests
4. Config fallback tests

### Phase 2: Feature Coverage (Week 2)
1. Quality filtering tests
2. Deduplication tests
3. Experience replay tests
4. UI training interface tests

### Phase 3: Edge Cases (Week 3)
1. Perplexity filtering tests
2. Streaming dataset tests
3. Curriculum learning tests
4. GPU monitoring UI tests

### Phase 4: Polish (Week 4)
1. File loading edge cases
2. Windows-specific tests
3. UI export interface tests
4. Statistics computation tests

---

## Test Utilities Needed

```python
# tests/test_helpers.py additions

class MockGradioInterface:
    """Mock Gradio interface for UI testing."""
    pass

class MockHuggingFaceHub:
    """Mock HuggingFace Hub for dataset tests."""
    pass

class MockPandasDataFrame:
    """Mock pandas for parquet/csv tests."""
    pass

class MockPerplexityModel:
    """Mock GPT-2 for perplexity tests."""
    pass
```

---

## Expected Coverage After Implementation

| Module | Before | After | Tests Added |
|--------|--------|-------|-------------|
| cli.py | 73% | 92% | ~35 |
| datasets.py | 68% | 87% | ~55 |
| config.py | 61% | 88% | ~25 |
| multi_run.py | 55% | 86% | ~45 |
| ui.py | 53% | 78% | ~30 |
| **Total** | **70%** | **86%** | **~190** |
