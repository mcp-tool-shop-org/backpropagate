"""Coverage tests for the pure helpers and branches of ``backpropagate.model_card``.

Nothing mocked except ``importlib.metadata.version`` (the package-version
lookup) and ``RunHistoryManager`` construction (to force its failure path).
"""

from __future__ import annotations

import importlib.metadata
import logging
from importlib.metadata import PackageNotFoundError

import pytest

from backpropagate import model_card as mc

# =============================================================================
# Sparkline
# =============================================================================


class TestBuildLossSparkline:
    @pytest.mark.parametrize("empty", [None, [], ["a", "b"], [None, "x"]])
    def test_nothing_numeric_renders_nothing(self, empty):
        assert mc.build_loss_sparkline(empty) == ""

    def test_flat_series_is_all_low_blocks(self):
        assert mc.build_loss_sparkline([0.5, 0.5, 0.5]) == mc._SPARKLINE_CHARS[0] * 3

    def test_descending_loss_goes_from_high_to_low_block(self):
        line = mc.build_loss_sparkline([4.0, 3.0, 2.0, 1.0, 0.0])
        assert len(line) == 5
        assert line[0] == mc._SPARKLINE_CHARS[-1] and line[-1] == mc._SPARKLINE_CHARS[0]
        indices = [mc._SPARKLINE_CHARS.index(c) for c in line]
        assert indices == sorted(indices, reverse=True)

    def test_non_numeric_entries_are_ignored(self):
        assert mc.build_loss_sparkline([1.0, "bad", 3.0]) == mc.build_loss_sparkline([1.0, 3.0])

    def test_long_runs_are_downsampled_to_width(self):
        series = [float(i) for i in range(1000)]
        line = mc.build_loss_sparkline(series, width=20)
        assert len(line) == 20
        assert line[0] == mc._SPARKLINE_CHARS[0] and line[-1] == mc._SPARKLINE_CHARS[-1]

    def test_series_at_exactly_width_is_not_downsampled(self):
        assert len(mc.build_loss_sparkline([1.0, 2.0, 3.0, 4.0], width=4)) == 4


# =============================================================================
# Small formatters / sanitisers
# =============================================================================


class TestShortName:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "Qwen2.5-7B-Instruct-bnb-4bit-finetune"),
            ("plain-name", "plain-name-finetune"),
            (None, "backpropagate-finetune"),
            ("", "backpropagate-finetune"),
            ("org/", "backpropagate-finetune"),
        ],
    )
    def test_names(self, base, expected):
        assert mc.infer_model_short_name(base) == expected


class TestFormatters:
    def test_format_value(self):
        assert mc._format_value(None) == "*(not recorded)*"
        assert mc._format_value(None, "n/a") == "n/a"
        assert mc._format_value(0.123456) == "0.1235"
        assert mc._format_value(7) == "7"
        assert mc._format_value("text") == "text"

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (None, "*(not recorded)*"), (0, "*(not recorded)*"), (-3, "*(not recorded)*"),
            ("12", "*(not recorded)*"), (12.34, "12.3 seconds"), (90, "1.5 minutes"),
            (3600, "1.00 hours"), (7200, "2.00 hours"),
        ],
    )
    def test_duration(self, seconds, expected):
        assert mc._format_duration(seconds) == expected

    def test_markdown_sanitiser_escapes_link_syntax(self):
        assert mc._sanitize_markdown(None) == ""
        assert mc._sanitize_markdown("plain") == "plain"
        out = mc._sanitize_markdown("evil.jsonl ](javascript:alert(1))")
        assert "](" not in out
        assert out == "evil.jsonl \\]\\(javascript:alert\\(1\\)\\)"
        assert mc._sanitize_markdown("a`b<c>\\d") == "a\\`b\\<c\\>\\\\d"

    def test_codespan_sanitiser(self):
        assert mc._sanitize_codespan(None) == ""
        assert mc._sanitize_codespan("") == ""
        assert mc._sanitize_codespan("a`b\nc\rd") == "aˋb c d"

    def test_yaml_scalar_sanitiser(self):
        assert mc._sanitize_yaml_scalar(None) == '""'
        assert mc._sanitize_yaml_scalar("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"
        assert mc._sanitize_yaml_scalar("llm") == "llm"
        assert mc._sanitize_yaml_scalar("") == '""'
        assert mc._sanitize_yaml_scalar("user:tag") == '"user:tag"'  # ':' forces quoting
        assert mc._sanitize_yaml_scalar('a "b"\nc\\d\re') == '"a \\"b\\"\\nc\\\\d\\re"'


# =============================================================================
# generate_model_card branches
# =============================================================================


class TestGenerateModelCard:
    def test_defaults_fill_created_timestamp_and_installed_version(self):
        card = mc.generate_model_card(run_id="r1", base_model="org/base")
        assert f"backpropagate=={importlib.metadata.version('backpropagate')}" in card
        created = [ln for ln in card.splitlines() if ln.startswith("| Created |")][0]
        assert created.count("-") >= 2 and "T" in created  # an ISO timestamp

    def test_unknown_version_when_metadata_is_missing(self, monkeypatch):
        def missing(name):
            raise PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", missing)
        card = mc.generate_model_card(base_model="org/base")
        assert "| Library version | `backpropagate==unknown` |" in card

    def test_dataset_variants(self):
        both = mc.generate_model_card(dataset_path="data.jsonl", dataset_hash="abc123")
        assert "| Dataset | `data.jsonl` (sha256: `abc123`) |" in both
        path_only = mc.generate_model_card(dataset_path="data.jsonl")
        assert "| Dataset | `data.jsonl` |" in path_only
        hash_only = mc.generate_model_card(dataset_hash="abc123")
        assert "| Dataset | (remote) sha256: `abc123` |" in hash_only
        neither = mc.generate_model_card()
        assert "| Dataset | *(not recorded)* |" in neither

    def test_orpo_run_is_tagged_and_shows_the_objective(self):
        with_beta = mc.generate_model_card(
            base_model="org/base", extra_hyperparameters={"method": "ORPO", "orpo_beta": 0.1}
        )
        assert "  - orpo" in with_beta
        assert "| Method | `orpo` (beta=0.1000) |" in with_beta
        no_beta = mc.generate_model_card(base_model="org/base", extra_hyperparameters={"method": "orpo"})
        assert "| Method | `orpo` |" in no_beta
        sft = mc.generate_model_card(base_model="org/base", extra_hyperparameters={"method": "sft"})
        assert "| Method |" not in sft and "  - orpo" not in sft

    def test_tags_quantization_and_extra_tags(self):
        card = mc.generate_model_card(
            quantization="q4_k_m", export_format="gguf", extra_tags=["custom", "", "llm", "custom"]
        )
        front = card.split("---")[1]
        assert front.count("  - custom") == 1  # de-duplicated, blanks and repeats dropped
        assert "  - gguf" in front and front.count("  - llm") == 1
        assert "| Export format | `gguf` |" in card and "| Quantization | `q4_k_m` |" in card

    def test_frontmatter_survives_hostile_base_model(self):
        card = mc.generate_model_card(base_model='evil: "x"\ninjected: true')
        front = card.split("---")[1]
        assert 'base_model: "evil: \\"x\\"\\ninjected: true"' in front
        assert "\ninjected: true" not in front

    def test_loss_curve_section_variants(self):
        with_curve = mc.generate_model_card(loss_history=[2.0, 1.0, 0.5])
        assert "## Loss curve" in with_curve and "(3 loss samples;" in with_curve
        empty = mc.generate_model_card(loss_history=[])
        assert "*(no loss samples recorded)*" in empty
        absent = mc.generate_model_card(loss_history=None)
        assert "## Loss curve" not in absent

    def test_incomplete_provenance_banner_and_reproduce_block(self):
        card = mc.generate_model_card(base_model="org/base", incomplete_provenance=True, steps=100, lora_r=16)
        assert "Incomplete provenance" in card
        assert "```bash" in card and "org/base" in card
        no_base = mc.generate_model_card()
        assert "```bash" not in no_base and "Fine-tuned via [backpropagate]" in no_base
        assert "Fine-tuned `org/base`" in card


class TestWriteAndLoad:
    def test_write_creates_directories_and_honours_filename(self, tmp_path):
        out = tmp_path / "deep" / "dir"
        path = mc.write_model_card_for_export(out, run_id="r9", base_model="org/base", filename="CARD.md")
        assert path == out / "CARD.md"
        assert "r9" in path.read_text(encoding="utf-8")

    def test_load_run_history_without_run_id(self, tmp_path):
        assert mc.load_run_history_for_card(tmp_path, None) is None
        assert mc.load_run_history_for_card(tmp_path, "") is None

    def test_load_run_history_unknown_run(self, tmp_path):
        assert mc.load_run_history_for_card(tmp_path, "nope") is None

    def test_load_run_history_returns_the_recorded_entry(self, tmp_path):
        from backpropagate.checkpoints import RunHistoryManager

        RunHistoryManager(str(tmp_path)).record_run_started(run_id="abc123", model_name="org/m")
        entry = mc.load_run_history_for_card(tmp_path, "abc")  # prefix match
        assert entry["run_id"] == "abc123" and entry["model_name"] == "org/m"

    def test_load_failure_is_swallowed_with_a_debug_log(self, tmp_path, monkeypatch, caplog):
        def boom(*a, **k):
            raise RuntimeError("history corrupt")

        monkeypatch.setattr("backpropagate.checkpoints.RunHistoryManager", boom)
        with caplog.at_level(logging.DEBUG, logger="backpropagate.model_card"):
            assert mc.load_run_history_for_card(tmp_path, "abc") is None
        assert any("load_run_history_for_card failed: history corrupt" in r.getMessage() for r in caplog.records)
