"""F-004: tests for the model card generator (backpropagate.model_card)."""

from pathlib import Path


class TestSparkline:
    """build_loss_sparkline behaviour."""

    def test_empty_returns_empty_string(self):
        from backpropagate.model_card import build_loss_sparkline

        assert build_loss_sparkline(None) == ""
        assert build_loss_sparkline([]) == ""

    def test_filters_non_numeric(self):
        from backpropagate.model_card import build_loss_sparkline

        sparkline = build_loss_sparkline([1.0, "skipme", 0.5, None, 0.1])  # type: ignore
        # only 1.0, 0.5, 0.1 contribute => 3 chars
        assert len(sparkline) == 3

    def test_flat_loss_renders_low_blocks(self):
        from backpropagate.model_card import build_loss_sparkline

        sparkline = build_loss_sparkline([0.5, 0.5, 0.5])
        assert all(ch == "▁" for ch in sparkline)  # low block

    def test_downsamples_to_width(self):
        from backpropagate.model_card import build_loss_sparkline

        sparkline = build_loss_sparkline(list(range(1000, 0, -1)), width=20)
        assert len(sparkline) == 20

    def test_descent_renders_decreasing_blocks(self):
        from backpropagate.model_card import build_loss_sparkline

        sparkline = build_loss_sparkline([1.0, 0.75, 0.5, 0.25, 0.0])
        # Strictly descending values map to descending block heights.
        # Block ordering goes from low(▁) to high(█); first char is highest.
        assert sparkline[0] == "█"  # highest
        assert sparkline[-1] == "▁"  # lowest


class TestInferModelShortName:

    def test_returns_finetune_suffix(self):
        from backpropagate.model_card import infer_model_short_name

        assert (
            infer_model_short_name("Qwen/Qwen2.5-7B-Instruct")
            == "Qwen2.5-7B-Instruct-finetune"
        )

    def test_none_returns_default(self):
        from backpropagate.model_card import infer_model_short_name

        assert infer_model_short_name(None) == "backpropagate-finetune"

    def test_empty_returns_default(self):
        from backpropagate.model_card import infer_model_short_name

        assert infer_model_short_name("") == "backpropagate-finetune"


class TestGenerateModelCard:

    def test_frontmatter_present(self):
        from backpropagate.model_card import generate_model_card

        card = generate_model_card(
            run_id="abc123",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            dataset_path="data.jsonl",
            final_loss=0.42,
            steps=100,
        )

        assert card.startswith("---\n")
        assert "library_name: backpropagate" in card
        assert "base_model: Qwen/Qwen2.5-7B-Instruct" in card
        assert "tags:" in card
        assert "  - llm" in card

    def test_property_table_renders_known_fields(self):
        from backpropagate.model_card import generate_model_card

        card = generate_model_card(
            run_id="abc123",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            dataset_path="data.jsonl",
            dataset_hash="deadbeefdeadbeef",
            final_loss=0.42,
            steps=100,
            lora_r=16,
            lora_alpha=32,
            seed=3407,
            training_duration=120.0,
            gpu_used="RTX 5080",
        )

        assert "`abc123`" in card
        assert "`Qwen/Qwen2.5-7B-Instruct`" in card
        assert "deadbeefdeadbeef" in card
        assert "0.4200" in card
        assert "| Steps | 100 |" in card
        assert "| LoRA rank | 16 |" in card
        assert "| Seed | 3407 |" in card
        assert "RTX 5080" in card
        assert "2.0 minutes" in card

    def test_missing_fields_render_placeholder(self):
        from backpropagate.model_card import generate_model_card

        card = generate_model_card(run_id=None, base_model=None)
        assert "*(not recorded)*" in card

    def test_loss_curve_only_when_present(self):
        from backpropagate.model_card import generate_model_card

        card_no = generate_model_card(run_id="x", loss_history=None)
        assert "Loss curve" not in card_no

        card_yes = generate_model_card(run_id="x", loss_history=[1.0, 0.5, 0.1])
        assert "Loss curve" in card_yes

    def test_quantization_tag_added_for_gguf(self):
        from backpropagate.model_card import generate_model_card

        card = generate_model_card(
            run_id="x",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            quantization="q4_k_m",
            export_format="gguf",
        )
        assert "  - gguf" in card
        assert "q4_k_m" in card

    def test_incomplete_provenance_banner(self):
        from backpropagate.model_card import generate_model_card

        card = generate_model_card(
            run_id="x",
            incomplete_provenance=True,
        )
        assert "Incomplete provenance" in card

    def test_reproduce_block_present_when_base_model_known(self):
        from backpropagate.model_card import generate_model_card

        card = generate_model_card(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            steps=50,
            lora_r=8,
        )
        assert "## Reproduce" in card
        assert "backprop train" in card
        assert "--model Qwen/Qwen2.5-7B-Instruct" in card
        assert "--steps 50" in card
        assert "--lora-r 8" in card

    def test_reproduce_block_absent_without_base_model(self):
        from backpropagate.model_card import generate_model_card

        card = generate_model_card(steps=50)
        assert "## Reproduce" not in card

    def test_extra_tags_deduplicated(self):
        from backpropagate.model_card import generate_model_card

        card = generate_model_card(
            base_model="m",
            extra_tags=["llm", "experimental"],
        )
        assert card.count("  - llm\n") == 1
        assert "  - experimental" in card

    def test_trust_signals_block_present(self):
        from backpropagate.model_card import generate_model_card

        card = generate_model_card(base_model="m")
        assert "Trust signals" in card
        assert "Sigstore" in card
        assert "run_id" in card

    def test_duration_formatting(self):
        from backpropagate.model_card import generate_model_card

        assert "1.5 seconds" in generate_model_card(training_duration=1.5)
        assert "5.0 minutes" in generate_model_card(training_duration=300)
        assert "1.50 hours" in generate_model_card(training_duration=5400)


class TestWriteModelCardForExport:

    def test_writes_file_at_path(self, tmp_path: Path):
        from backpropagate.model_card import write_model_card_for_export

        out = write_model_card_for_export(
            tmp_path,
            run_id="abc",
            base_model="m",
            final_loss=0.1,
            steps=10,
        )

        assert out.exists()
        assert out.name == "model_card.md"
        assert "abc" in out.read_text(encoding="utf-8")

    def test_custom_filename(self, tmp_path: Path):
        from backpropagate.model_card import write_model_card_for_export

        out = write_model_card_for_export(
            tmp_path,
            run_id="abc",
            filename="README.md",
        )
        assert out.name == "README.md"

    def test_creates_parent_directory(self, tmp_path: Path):
        from backpropagate.model_card import write_model_card_for_export

        target = tmp_path / "nested" / "dirs"
        out = write_model_card_for_export(target, run_id="abc")
        assert out.parent == target
        assert out.exists()


class TestLoadRunHistoryForCard:

    def test_returns_none_for_missing_run_id(self, tmp_path: Path):
        from backpropagate.model_card import load_run_history_for_card

        assert load_run_history_for_card(tmp_path, None) is None
        assert load_run_history_for_card(tmp_path, "missing") is None

    def test_resolves_run_record(self, tmp_path: Path):
        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.model_card import load_run_history_for_card

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(run_id="present", model_name="m")
        manager.record_run_completed(run_id="present", final_loss=0.1)

        record = load_run_history_for_card(tmp_path, "present")
        assert record is not None
        assert record["run_id"] == "present"
        assert record["final_loss"] == 0.1


class TestPackageExports:

    def test_generate_model_card_reexported(self):
        import backpropagate

        assert hasattr(backpropagate, "generate_model_card")
        assert hasattr(backpropagate, "write_model_card")
