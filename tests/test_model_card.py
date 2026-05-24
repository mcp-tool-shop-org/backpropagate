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


# =============================================================================
# BACKEND-F-016 — reproduce block reflects captured hyperparameters
# =============================================================================
#
# Cross-domain pin from Wave 5.5 BACKEND-F-016 fix
# (backpropagate/model_card.py::_build_reproduce_block). Pre-fix, the
# Reproduce block in the generated model card hardcoded the
# ``backprop train`` invocation with only --model / --data / --steps /
# --lora-r, omitting --batch-size / --lr / --max-seq-length /
# --no-unsloth / --lora-alpha / --seed. An operator who copy-pasted
# the printed command did NOT reproduce the model — the dropped flags
# defaulted back to whatever the running CLI had, and the reproduce
# claim was a documented lie.
#
# Multi-run sessions had the same single-run shape, doubly wrong:
# the operator's actual invocation was
# ``backprop multi-run --runs N --steps-per-run S --merge-mode ...``,
# not ``backprop train``. The reproduce block printed a command that
# would have created a single-run model with a default merge config —
# fundamentally different output.
#
# The fix routes ``session_kind`` + ``extra_hyperparameters`` into the
# renderer so every captured runtime value lands in the printed
# command. This test class pins:
#   - single-run shape carries batch_size / lr / max_seq_length /
#     lora_alpha / seed / use_unsloth=False → --no-unsloth
#   - multi-run shape carries runs / steps-per-run / samples-per-run /
#     merge-mode / initial-lr / final-lr / lr-decay / warmup
class TestReproduceBlockHyperparameters:
    """BACKEND-F-016 regression: reproduce block reflects all captured hyperparams."""

    def test_single_run_reproduce_block_emits_all_load_bearing_flags(self):
        """A single-run card built with non-default hyperparameters
        produces a Reproduce command containing every value the
        operator passed. This is the load-bearing F-016 contract: the
        printed command must reproduce the model.

        Coverage:
          - --batch-size from extra_hyperparameters['batch_size']
          - --lr from extra_hyperparameters['learning_rate']
          - --max-seq-length from extra_hyperparameters['max_seq_length']
          - --lora-alpha (lora_alpha overrides extra_hyperparameters)
          - --no-unsloth when use_unsloth=False
          - --seed
        """
        from backpropagate.model_card import generate_model_card

        # Non-default values for every load-bearing knob so a printed
        # default would obviously fail the assertion (a printed "1024"
        # for max_seq_length would land on the default and we'd miss
        # the regression).
        batch_size_value = 7
        lr_value = 3.7e-5
        max_seq_length_value = 4321
        lora_alpha_value = 99
        seed_value = 13579

        card = generate_model_card(
            run_id="f016-single-run-pin",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            dataset_path="/data/hyperparam-pin.jsonl",
            steps=137,
            lora_r=24,
            lora_alpha=lora_alpha_value,
            seed=seed_value,
            session_kind="single_run",
            extra_hyperparameters={
                "batch_size": batch_size_value,
                "learning_rate": lr_value,
                "max_seq_length": max_seq_length_value,
                "lora_alpha": lora_alpha_value,
                "use_unsloth": False,
                "seed": seed_value,
            },
        )

        # The Reproduce block lives inside a ```bash code-fence; we
        # extract it so we don't accidentally match on a value
        # appearing in the property table or the trust signals
        # section above.
        reproduce_block = _extract_reproduce_block(card)
        assert reproduce_block, (
            "BACKEND-F-016: Reproduce block missing from generated "
            "card despite base_model being supplied."
        )

        # Subcommand: single-run uses `backprop train`.
        assert "backprop train" in reproduce_block, (
            f"single-run session_kind should emit `backprop train`. "
            f"Got: {reproduce_block!r}"
        )

        # Model + dataset (the pre-F-016 baseline; we still pin them
        # so a refactor doesn't drop them).
        assert "Qwen/Qwen2.5-7B-Instruct" in reproduce_block
        assert "/data/hyperparam-pin.jsonl" in reproduce_block

        # The F-016 additions: each load-bearing flag MUST appear.
        # The value rendering is via repr() for floats and str() for
        # ints, so we check the formatted string each flag would emit.
        assert "--steps 137" in reproduce_block
        assert f"--batch-size {batch_size_value}" in reproduce_block, (
            f"BACKEND-F-016 contract violation: --batch-size missing "
            f"or wrong value. Block: {reproduce_block!r}"
        )
        # repr(3.7e-5) is platform-stable enough to match exactly.
        assert f"--lr {lr_value!r}" in reproduce_block, (
            f"BACKEND-F-016 contract violation: --lr missing or wrong "
            f"value. Block: {reproduce_block!r}"
        )
        assert f"--max-seq-length {max_seq_length_value}" in reproduce_block, (
            f"BACKEND-F-016 contract violation: --max-seq-length missing "
            f"or wrong value. Block: {reproduce_block!r}"
        )
        assert f"--lora-alpha {lora_alpha_value}" in reproduce_block, (
            f"BACKEND-F-016 contract violation: --lora-alpha missing "
            f"or wrong value. Block: {reproduce_block!r}"
        )
        assert "--no-unsloth" in reproduce_block, (
            f"BACKEND-F-016 contract violation: --no-unsloth missing "
            f"despite use_unsloth=False. The reproduce command would "
            f"silently re-enable Unsloth. Block: {reproduce_block!r}"
        )
        assert f"--seed {seed_value}" in reproduce_block, (
            f"BACKEND-F-016 contract violation: --seed missing or "
            f"wrong value. Block: {reproduce_block!r}"
        )

    def test_single_run_omits_no_unsloth_when_use_unsloth_default(self):
        """``--no-unsloth`` is the right flag when the run was trained
        WITHOUT unsloth (operator opt-out). Absent / True value
        defaults to unsloth-enabled (the standard path) so the flag
        must NOT appear. Pin both directions so a future refactor
        that adds --no-unsloth unconditionally is caught.
        """
        from backpropagate.model_card import generate_model_card

        for use_unsloth_value in (True, None):
            extra = {
                "batch_size": 2,
                "learning_rate": 2e-4,
                "max_seq_length": 2048,
            }
            if use_unsloth_value is not None:
                extra["use_unsloth"] = use_unsloth_value

            card = generate_model_card(
                base_model="m",
                steps=10,
                session_kind="single_run",
                extra_hyperparameters=extra,
            )
            reproduce_block = _extract_reproduce_block(card)
            assert "--no-unsloth" not in reproduce_block, (
                f"BACKEND-F-016: --no-unsloth must NOT appear when "
                f"use_unsloth={use_unsloth_value!r}. Block: "
                f"{reproduce_block!r}"
            )

    def test_multi_run_reproduce_block_emits_multi_run_shape(self):
        """A multi-run card produces ``backprop multi-run`` (not
        ``backprop train``) and emits every load-bearing multi-run
        flag from the captured hyperparameters.

        Coverage:
          - subcommand changes to ``backprop multi-run``
          - --runs / --steps-per-run / --samples-per-run
          - --merge-mode
          - --initial-lr / --final-lr / --lr-decay
          - --warmup-steps-per-run
          - --seed (inherited from the kwarg if not in
            extra_hyperparameters)
        """
        from backpropagate.model_card import generate_model_card

        num_runs_value = 8
        steps_per_run_value = 137
        samples_per_run_value = 543
        merge_mode_value = "slao"
        initial_lr_value = 5.5e-4
        final_lr_value = 7.7e-5
        lr_decay_value = "cosine"
        warmup_value = 17
        seed_value = 24680

        card = generate_model_card(
            run_id="f016-multi-run-pin",
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            dataset_path="/data/multi-run-pin.jsonl",
            steps=None,  # multi-run uses steps_per_run, not steps
            session_kind="multi_run",
            seed=seed_value,
            extra_hyperparameters={
                "num_runs": num_runs_value,
                "steps_per_run": steps_per_run_value,
                "samples_per_run": samples_per_run_value,
                "merge_mode": merge_mode_value,
                "initial_lr": initial_lr_value,
                "final_lr": final_lr_value,
                "lr_decay": lr_decay_value,
                "warmup_steps_per_run": warmup_value,
            },
        )

        reproduce_block = _extract_reproduce_block(card)
        assert reproduce_block, (
            "BACKEND-F-016 multi-run: Reproduce block missing from "
            "generated card despite base_model being supplied."
        )

        # Subcommand: multi-run uses `backprop multi-run`.
        assert "backprop multi-run" in reproduce_block, (
            f"BACKEND-F-016 contract violation: multi-run session_kind "
            f"should emit `backprop multi-run`, not `backprop train`. "
            f"Block: {reproduce_block!r}"
        )
        # The single-run subcommand MUST NOT appear (pin so a future
        # refactor that emits both shapes is caught).
        assert "backprop train" not in reproduce_block, (
            f"BACKEND-F-016 contract violation: multi-run reproduce "
            f"block leaked `backprop train`. Block: {reproduce_block!r}"
        )

        # Model + dataset.
        assert "meta-llama/Llama-3.2-3B-Instruct" in reproduce_block
        assert "/data/multi-run-pin.jsonl" in reproduce_block

        # Multi-run flags.
        assert f"--runs {num_runs_value}" in reproduce_block, (
            f"--runs missing or wrong value. Block: {reproduce_block!r}"
        )
        assert f"--steps-per-run {steps_per_run_value}" in reproduce_block, (
            f"--steps-per-run missing or wrong value. Block: {reproduce_block!r}"
        )
        assert f"--samples-per-run {samples_per_run_value}" in reproduce_block, (
            f"--samples-per-run missing or wrong value. Block: {reproduce_block!r}"
        )
        assert f"--merge-mode {merge_mode_value}" in reproduce_block, (
            f"--merge-mode missing or wrong value. Block: {reproduce_block!r}"
        )
        assert f"--initial-lr {initial_lr_value!r}" in reproduce_block, (
            f"--initial-lr missing or wrong value. Block: {reproduce_block!r}"
        )
        assert f"--final-lr {final_lr_value!r}" in reproduce_block, (
            f"--final-lr missing or wrong value. Block: {reproduce_block!r}"
        )
        assert f"--lr-decay {lr_decay_value}" in reproduce_block, (
            f"--lr-decay missing or wrong value. Block: {reproduce_block!r}"
        )
        assert f"--warmup-steps-per-run {warmup_value}" in reproduce_block, (
            f"--warmup-steps-per-run missing or wrong value. "
            f"Block: {reproduce_block!r}"
        )
        assert f"--seed {seed_value}" in reproduce_block, (
            f"--seed missing or wrong value. Block: {reproduce_block!r}"
        )

    def test_multi_run_missing_values_render_as_placeholders(self):
        """Per F-016 design: missing values render as literal
        ``<placeholder>`` so the operator immediately sees which knobs
        are missing rather than getting a plausible-looking command
        that silently lies about defaults.
        """
        from backpropagate.model_card import generate_model_card

        # Multi-run shape with NO extra_hyperparameters — every knob
        # should fall through to its placeholder.
        card = generate_model_card(
            base_model="m",
            session_kind="multi_run",
            extra_hyperparameters={},
        )
        block = _extract_reproduce_block(card)
        for expected in (
            "<runs>",
            "<steps-per-run>",
            "<samples-per-run>",
            "<merge-mode>",
            "<initial-lr>",
            "<final-lr>",
            "<lr-decay>",
            "<warmup-steps-per-run>",
            "<seed>",
        ):
            assert expected in block, (
                f"BACKEND-F-016: placeholder {expected!r} missing — a "
                f"plausible default leaked instead. Block: {block!r}"
            )


def _extract_reproduce_block(card: str) -> str:
    """Helper: pull the ```bash ... ``` Reproduce code-fence out of a
    rendered model card so assertions don't accidentally match values
    appearing in the property table or trust-signals section.
    """
    marker = "## Reproduce"
    if marker not in card:
        return ""
    after = card.split(marker, 1)[1]
    if "```bash" not in after:
        return ""
    fence = after.split("```bash", 1)[1]
    return fence.split("```", 1)[0]
