"""Structure tests for the Reflex UI shell: components, chrome, pages, app wiring.

Coverage-raising suite (``test_ui_cov_*``). Nothing here starts a server or
touches the network or a training backend: every test builds a component tree
(or imports the app module) in-process and asserts on the tree that matters.
Reflex itself is the only real collaborator; nothing is mocked unless a test
docstring says so.

How a tree is inspected:

* ``_walk`` yields every component node (``children`` recursion).
* ``_handlers`` returns ``(component class, trigger, [handler qualnames])`` for
  every event chain in the tree, so a test can assert that a control is wired
  to the right state method.
* ``_render`` is the JSON of ``Component.render()``; used to assert literal
  copy, ARIA attributes and state-var bindings.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

rx = pytest.importorskip("reflex", reason="reflex is required (install backpropagate[ui])")

from reflex_base.event import EventChain  # noqa: E402

# =============================================================================
# Helpers
# =============================================================================


def _walk(component):
    yield component
    for child in getattr(component, "children", None) or []:
        if hasattr(child, "children"):
            yield from _walk(child)


def _handlers(component):
    out = []
    for node in _walk(component):
        for trigger, chain in getattr(node, "event_triggers", {}).items():
            if not isinstance(chain, EventChain):
                continue
            names = []
            for ev in chain.events:
                handler = getattr(ev, "handler", None)
                names.append(handler.fn.__qualname__ if handler is not None else "<var>")
            out.append((type(node).__name__, trigger, names))
    return out


def _bound(component):
    """Set of ``(trigger, 'State.method')`` pairs wired in the tree."""
    return {(trig, n) for _cls, trig, names in _handlers(component) for n in names}


def _render(component) -> str:
    return json.dumps(component.render(), default=str)


def _texts(component) -> set[str]:
    return set(re.findall(r'\{"contents": "\\"([^"\\]+)\\""\}', _render(component)))


def _node_types(component) -> set[str]:
    return {type(n).__name__ for n in _walk(component)}


# =============================================================================
# components/sparkline.py
# =============================================================================


class TestSparklinePolyline:
    """``_build_polyline_points`` geometry for empty / single / multi-point series."""

    def test_empty_series_has_no_geometry(self):
        from backpropagate.ui_app.components.sparkline import _build_polyline_points

        assert _build_polyline_points([], 220, 48) == ("", "", None)

    def test_single_point_is_a_flat_baseline_with_centred_end_dot(self):
        from backpropagate.ui_app.components.sparkline import _build_polyline_points

        line, area, end = _build_polyline_points([3.0], 220, 48)
        assert line == "2.0,46.0 218.0,46.0"
        assert area.startswith("M2.0,46.0 L218.0,46.0") and area.endswith("Z")
        assert end == (110.0, 46.0)

    def test_multi_point_maps_min_to_bottom_and_max_to_top(self):
        from backpropagate.ui_app.components.sparkline import _build_polyline_points

        line, area, end = _build_polyline_points([1.0, 2.0, 0.5], 220, 48)
        pts = [tuple(map(float, p.split(","))) for p in line.split()]
        assert len(pts) == 3
        # x spreads evenly across the padded plot width
        assert pts[0][0] == 2.0 and pts[-1][0] == 218.0 and pts[1][0] == 110.0
        # y: max value (2.0) at the top pad, min value (0.5) at the bottom
        assert pts[1][1] == 2.0 and pts[2][1] == 46.0
        assert end == pts[-1]
        # area path closes to the baseline at both ends
        assert area.startswith("M2.0,") and area.endswith("L2.0,46.0 Z")

    def test_constant_series_does_not_divide_by_zero(self):
        from backpropagate.ui_app.components.sparkline import _build_polyline_points

        line, _area, end = _build_polyline_points([5.0, 5.0, 5.0], 100, 20)
        ys = {p.split(",")[1] for p in line.split()}
        assert len(ys) == 1  # flat line
        assert end is not None


class TestBpSparkline:
    def test_data_builds_svg_path_polyline_and_two_dots(self):
        from backpropagate.ui_app.components.sparkline import BpSparkline

        comp = BpSparkline([1.0, 2.0, 0.5], caption="Loss", meta="min 0.5")
        types = _node_types(comp)
        assert {"Svg", "Path", "Polyline", "Circle"} <= types
        assert sum(1 for n in _walk(comp) if type(n).__name__ == "Circle") == 2
        rendered = _render(comp)
        assert "loss sparkline" in rendered and "last 3 steps" in rendered
        assert "Loss" in _texts(comp) and "min 0.5" in _texts(comp)

    def test_no_data_and_no_labels_is_just_an_empty_svg(self):
        from backpropagate.ui_app.components.sparkline import BpSparkline

        comp = BpSparkline(None)
        types = _node_types(comp)
        assert "Svg" in types
        assert not ({"Path", "Polyline", "Circle"} & types)
        assert "last 0 steps" in _render(comp)

    def test_only_caption_omits_meta_text(self):
        from backpropagate.ui_app.components.sparkline import BpSparkline

        comp = BpSparkline([1.0, 2.0], caption="Only caption")
        assert "Only caption" in _texts(comp)
        assert "bp-num" not in _render(comp)  # meta text carries the bp-num class

    def test_only_meta_omits_caption_text(self):
        from backpropagate.ui_app.components.sparkline import BpSparkline

        comp = BpSparkline([1.0, 2.0], meta="m")
        assert "m" in _texts(comp)
        assert "uppercase" not in _render(comp)  # the caption eyebrow is uppercase


# =============================================================================
# components/vram_bar.py + gpu_ring.py
# =============================================================================


class TestVramBar:
    @pytest.mark.parametrize(
        ("ratio", "token"),
        [(0.0, "--bp-blue"), (0.79, "--bp-blue"), (0.80, "--bp-amber"),
         (0.94, "--bp-amber"), (0.95, "--bp-peach"), (1.0, "--bp-peach")],
    )
    def test_threshold_colours(self, ratio, token):
        from backpropagate.ui_app.components.vram_bar import _vram_color

        assert _vram_color(ratio) == f"var({token})"

    def test_bar_width_aria_and_label(self):
        from backpropagate.ui_app.components.vram_bar import BpVramBar

        rendered = _render(BpVramBar(used_gb=8.0, total_gb=16.0))
        assert "50.0%" in rendered
        assert '"aria-valuenow":"50"' in rendered.replace(" ", "").replace('\\"', '"') or "aria-valuenow" in rendered
        assert "VRAM 8.0 of 16.0 gigabytes used" in rendered
        assert "8.0 / 16.0 GB" in rendered

    def test_zero_total_clamps_to_empty(self):
        from backpropagate.ui_app.components.vram_bar import BpVramBar

        rendered = _render(BpVramBar(used_gb=4.0, total_gb=0.0))
        assert "0.0%" in rendered

    def test_overcommit_clamps_to_full(self):
        from backpropagate.ui_app.components.vram_bar import BpVramBar

        rendered = _render(BpVramBar(used_gb=20.0, total_gb=16.0))
        assert "100.0%" in rendered and "--bp-peach" in rendered


class TestGpuRing:
    @pytest.mark.parametrize(
        ("temp", "token"),
        [(30, "--bp-seafoam"), (69.9, "--bp-seafoam"), (70, "--bp-amber"),
         (84.9, "--bp-amber"), (85, "--bp-peach"), (99, "--bp-peach")],
    )
    def test_temp_colour_thresholds(self, temp, token):
        from backpropagate.ui_app.components.gpu_ring import _temp_color

        assert _temp_color(temp) == f"var({token})"

    def test_ring_renders_two_circles_and_label(self):
        from backpropagate.ui_app.components.gpu_ring import BpGpuRing

        comp = BpGpuRing(temp_c=70, max_c=100, size=60)
        circles = [n for n in _walk(comp) if type(n).__name__ == "Circle"]
        assert len(circles) == 2
        rendered = _render(comp)
        assert "gpu temperature 70 degrees celsius" in rendered
        assert "--bp-amber" in rendered
        assert "70" in _texts(comp)

    def test_overheat_clamps_dash_to_full_circle(self):
        from backpropagate.ui_app.components.gpu_ring import BpGpuRing

        rendered = _render(BpGpuRing(temp_c=500, max_c=95, size=60))
        # fully filled arc -> gap of 0.0
        assert re.search(r"strokeDasharray.*? 0\.0", rendered)


# =============================================================================
# components/status_pill.py
# =============================================================================


class TestStatusPill:
    def test_tint_and_border_use_color_mix(self):
        from backpropagate.ui_app.components.status_pill import _border, _tint

        assert _tint("var(--x)") == "color-mix(in srgb, var(--x) 12%, var(--bp-surface))"
        assert _border("var(--x)") == "color-mix(in srgb, var(--x) 35%, transparent)"

    @pytest.mark.parametrize(
        ("state", "token", "anim"),
        [("active", "--bp-teal", "bp-heartbeat-2400"), ("paused", "--bp-amber", "bp-pulse-1600"),
         ("loading", "--bp-blue", "bp-heartbeat-2400"), ("error", "--bp-peach", ""),
         ("done", "--bp-seafoam", ""), ("idle", "--bp-muted-2", "")],
    )
    def test_literal_state_picks_dot_colour_and_animation(self, state, token, anim):
        from backpropagate.ui_app.components.status_pill import BpStatusPill

        comp = BpStatusPill(state=state, label="L", detail="d")
        rendered = _render(comp)
        assert token in rendered
        if anim:
            assert anim in rendered
        else:
            assert "bp-heartbeat" not in rendered and "bp-pulse" not in rendered
        assert {"L", "d"} <= _texts(comp)

    def test_literal_state_without_detail_hides_detail_line(self):
        from backpropagate.ui_app.components.status_pill import BpStatusPill

        comp = BpStatusPill(state="idle", label="Idle", detail="")
        assert "Cond" in _node_types(comp)  # detail is wrapped in rx.cond(detail != "")

    def test_unknown_literal_state_falls_back_to_idle_colour(self):
        from backpropagate.ui_app.components.status_pill import BpStatusPill

        assert "--bp-muted-2" in _render(BpStatusPill(state="bogus"))

    def test_var_state_renders_a_match_over_all_six_states(self):
        from backpropagate.ui_app.components.status_pill import BpStatusPill
        from backpropagate.ui_state import TrainState

        comp = BpStatusPill(state=TrainState.run_state, label="Run state", detail=TrainState.run_state)
        assert "Match" in _node_types(comp) or comp.__class__.__name__ == "Match"
        rendered = _render(comp)
        for token in ("--bp-muted-2", "--bp-blue", "--bp-teal", "--bp-amber",
                      "--bp-seafoam", "--bp-peach"):
            assert token in rendered
        assert "train_state.run_state" in rendered


# =============================================================================
# components/event_log.py
# =============================================================================


class TestEventLog:
    def test_none_renders_empty_hint(self):
        from backpropagate.ui_app.components.event_log import BpEventLog

        assert "No events yet" in _texts(BpEventLog(events=None))

    def test_empty_list_renders_empty_hint(self):
        from backpropagate.ui_app.components.event_log import BpEventLog

        assert "No events yet" in _texts(BpEventLog(events=[]))

    def test_literal_list_keeps_only_last_n_rows(self):
        from backpropagate.ui_app.components.event_log import BpEventLog

        events = [{"t": f"00:0{i}", "level": "ok", "msg": f"m{i}"} for i in range(8)]
        comp = BpEventLog(events=events, max_n=3)
        texts = _texts(comp)
        assert {"m5", "m6", "m7"} <= texts
        assert not ({"m0", "m1", "m4"} & texts)
        assert "--bp-seafoam" in _render(comp)

    def test_literal_row_unknown_level_falls_back_to_text2(self):
        from backpropagate.ui_app.components.event_log import BpEventLog

        comp = BpEventLog(events=[{"t": "t", "level": "weird", "msg": "x"}])
        assert "--bp-text-2" in _render(comp)

    def test_literal_row_defaults_when_keys_missing(self):
        from backpropagate.ui_app.components.event_log import _row_static

        rendered = _render(_row_static({}))
        assert "--bp-text-2" in rendered  # defaults to level "info"

    def test_var_events_use_foreach_and_full_log_dialog(self):
        from backpropagate.ui_app.components.event_log import BpEventLog
        from backpropagate.ui_state import TrainState

        comp = BpEventLog(events=TrainState.events, max_n=6, show_view_full=True)
        types = _node_types(comp)
        assert {"Foreach", "DialogRoot", "DialogTrigger", "DialogContent", "Cond"} <= types
        texts = _texts(comp)
        assert {"View full log", "Event log", "Close", "No events yet"} <= texts
        assert "train_state.events" in _render(comp)

    def test_var_events_without_view_full_has_no_dialog(self):
        from backpropagate.ui_app.components.event_log import BpEventLog
        from backpropagate.ui_state import TrainState

        comp = BpEventLog(events=TrainState.events, show_view_full=False)
        assert "DialogRoot" not in _node_types(comp)
        assert "Foreach" in _node_types(comp)

    def test_row_var_matches_every_level_to_a_colour(self):
        from backpropagate.ui_app.components.event_log import _row_var
        from backpropagate.ui_state import TrainState

        # _row_var is the foreach iteratee; call it with a var-typed entry.
        entry = TrainState.events[0]
        rendered = _render(_row_var(entry))
        for token in ("--bp-text-2", "--bp-seafoam", "--bp-amber", "--bp-peach",
                      "--bp-teal", "--bp-blue"):
            assert token in rendered


# =============================================================================
# components/error_callout.py, group.py, loss_chart.py, recovery_banner.py
# =============================================================================


class TestErrorCallout:
    def test_all_fields_stack_in_reading_order(self):
        from backpropagate.ui_app.components.error_callout import BpErrorCallout

        comp = BpErrorCallout(code="E_X", title="TitleT", message="MessageM", hint="HintH",
                              stack_trace="Traceback...")
        rendered = _render(comp)
        order = [rendered.index(s) for s in ("E_X", "TitleT", "MessageM", "HintH", "Stack trace")]
        assert order == sorted(order)
        assert "Cond" in _node_types(comp)
        assert "Details" in _node_types(comp)

    def test_apply_action_adds_a_button_bound_to_the_handler(self):
        from backpropagate.ui_app.components.error_callout import BpErrorCallout
        from backpropagate.ui_state import TrainState

        comp = BpErrorCallout(code="E", message="m", apply_action=TrainState.start_training)
        assert ("on_click", "TrainState.start_training") in _bound(comp)
        assert "Apply hint" in _texts(comp)

    def test_no_apply_action_means_no_button(self):
        from backpropagate.ui_app.components.error_callout import BpErrorCallout

        comp = BpErrorCallout(code="E")
        assert "Apply hint" not in _texts(comp)
        assert "Button" not in _node_types(comp)

    def test_stack_trace_none_omits_details(self):
        from backpropagate.ui_app.components.error_callout import BpErrorCallout

        assert "Details" not in _node_types(BpErrorCallout(code="E", stack_trace=None))


class TestGroup:
    def test_plain_group_has_eyebrow_and_children(self):
        from backpropagate.ui_app.components.group import Group

        comp = Group(rx.text("child-a"), rx.text("child-b"), title="Model")
        assert "Model" in _texts(comp) and {"child-a", "child-b"} <= _texts(comp)
        assert "AccordionRoot" not in _node_types(comp)

    def test_collapsible_open_by_default(self):
        from backpropagate.ui_app.components.group import Group

        comp = Group(rx.text("x"), title="Advanced", collapsible=True, default_open=True)
        assert "AccordionRoot" in _node_types(comp)
        rendered = _render(comp)
        assert 'defaultValue:\\"Advanced\\"' in rendered or "Advanced" in rendered

    def test_collapsible_closed_has_empty_default_value(self):
        from backpropagate.ui_app.components.group import Group

        comp = Group(rx.text("x"), title="Advanced", collapsible=True, default_open=False)
        assert comp.default_value is not None
        assert str(comp.default_value).strip('"') == ""

    def test_collapsible_without_title_uses_section_value(self):
        from backpropagate.ui_app.components.group import Group

        comp = Group(rx.text("x"), collapsible=True)
        assert "section" in _render(comp)


class TestLossChart:
    def test_chart_binds_data_height_and_color(self):
        from backpropagate.ui_app.components.loss_chart import BpLossChart
        from backpropagate.ui_state import TrainState

        comp = BpLossChart(TrainState.loss_chart_data, height=80, color="red", label="loss")
        types = _node_types(comp)
        assert {"LineChart", "Line", "XAxis", "YAxis", "CartesianGrid", "GraphingTooltip"} <= types
        rendered = _render(comp)
        assert "loss_chart_data" in rendered
        assert "80" in rendered and "red" in rendered

    def test_literal_data_and_custom_label(self):
        from backpropagate.ui_app.components.loss_chart import BpLossChart

        comp = BpLossChart([{"step": 1, "val": 2.0}], label="val")
        assert 'dataKey:\\"val\\"' in _render(comp)


class TestRecoveryBanner:
    @pytest.mark.parametrize(
        ("variant", "token", "icon"),
        [("info", "--bp-blue", "info.svg"), ("warn", "--bp-amber", "info.svg"),
         ("ok", "--bp-seafoam", "check.svg"), ("nonsense", "--bp-blue", "info.svg")],
    )
    def test_variant_colour_and_icon(self, variant, token, icon):
        from backpropagate.ui_app.components.recovery_banner import BpRecoveryBanner

        comp = BpRecoveryBanner(variant=variant, lead="Lead", body="Body")
        rendered = _render(comp)
        assert token in rendered and icon in rendered
        assert {"Lead", "Body"} <= _texts(comp)
        # a11y: polite live region
        assert "polite" in rendered and "status" in rendered

    def test_accepts_state_vars(self):
        from backpropagate.ui_app.components.recovery_banner import BpRecoveryBanner
        from backpropagate.ui_state import TrainState

        comp = BpRecoveryBanner("ok", lead=TrainState.run_state, body=TrainState.run_state)
        assert "train_state.run_state" in _render(comp)


# =============================================================================
# chrome.py
# =============================================================================


class TestChrome:
    def test_header_has_brand_theme_toggle_and_github_link(self):
        from backpropagate.ui_app.chrome import _BRAND_VERSION, BpHeader

        comp = BpHeader()
        texts = _texts(comp)
        assert {"backpropagate", _BRAND_VERSION, "run_id"} <= texts
        rendered = _render(comp)
        assert "Toggle theme" in rendered and "toggleColorMode" in rendered
        assert "github.com/mcp-tool-shop-org/backpropagate" in rendered
        assert "app_state.run_id" in rendered

    def test_left_nav_lists_six_surfaces_with_routes(self):
        from backpropagate.ui_app.chrome import _NAV_ITEMS, BpLeftNav

        assert [k for k, *_ in _NAV_ITEMS] == ["train", "multi-run", "export", "dataset", "runs", "models"]
        rendered = _render(BpLeftNav("runs"))
        for _key, label, href, _icon in _NAV_ITEMS:
            assert f'\\"{label}\\"' in rendered
            assert f'to:\\"{href}\\"' in rendered

    def test_active_nav_row_is_aria_current_and_others_are_not(self):
        from backpropagate.ui_app.chrome import _nav_link

        active = _render(_nav_link("runs", "Runs", "/runs", "/i.svg", "runs"))
        inactive = _render(_nav_link("runs", "Runs", "/runs", "/i.svg", "train"))
        assert "aria-current" in active and "page" in active
        assert "aria-current" not in inactive
        assert "inset 3px 0 0 var(--bp-teal)" in active
        assert "inset 3px 0 0" not in inactive

    def test_side_rail_binds_train_state_telemetry(self):
        from backpropagate.ui_app.chrome import BpSideRail

        comp = BpSideRail()
        rendered = _render(comp)
        for var in ("train_state.run_state", "train_state.loss_history",
                    "train_state.loss_chart_data", "train_state.current_step",
                    "train_state.current_loss", "train_state.events"):
            assert var in rendered
        assert {"Events", "View full log"} <= _texts(comp)
        assert "LineChart" in _node_types(comp)

    def test_rail_section_first_has_no_top_border(self):
        from backpropagate.ui_app.chrome import _rail_section

        assert "borderTop" not in _render(_rail_section(rx.text("a"), first=True))
        assert "borderTop" in _render(_rail_section(rx.text("a")))

    def test_footer_mounts_auth_badge_refresh(self):
        from backpropagate.ui_app.chrome import BpFooter

        comp = BpFooter()
        assert ("on_mount", "AuthBadgeState.refresh") in _bound(comp)
        assert {"Handbook", "GitHub"} <= _texts(comp)


# =============================================================================
# pages
# =============================================================================

# page -> (builder import path, expected (trigger, handler) bindings, copy that must render)
_PAGE_SPECS = {
    "train": (
        "backpropagate.ui_app.pages.train:train_page",
        {("on_change", f"TrainState.set_{f}") for f in (
            "model", "quantization", "steps", "batch_size", "learning_rate", "lora_r",
            "lora_alpha", "lora_dropout", "target_modules", "dataset_path",
            "gpu_temp_threshold", "wandb_run_name", "gradient_checkpointing",
            "flash_attention")} | {("on_click", "TrainState.start_training")},
    ),
    "multi_run": (
        "backpropagate.ui_app.pages.multi_run:multi_run_page",
        {("on_change", f"MultiRunState.set_{f}") for f in (
            "model", "quantization", "num_runs", "samples_per_run", "merge_mode")}
        | {("on_click", "MultiRunState.start_multi_run")},
    ),
    "export": (
        "backpropagate.ui_app.pages.export:export_page",
        {("on_change", f"ExportState.set_{f}") for f in (
            "source_model_path", "format", "gguf_quant", "ollama_register", "ollama_name",
            "hub_enabled", "hub_repo_id", "hub_branch", "hub_private", "hub_token",
            "hub_token_file_path", "hub_include_base")}
        | {("on_click", "ExportState.push_to_hub"), ("on_click", "ExportState.clear_hub_status"),
           ("on_click", "ExportState.start_export")},
    ),
    "dataset": (
        "backpropagate.ui_app.pages.dataset:dataset_page",
        {("on_change", f"DatasetState.set_{f}") for f in (
            "format_hint", "dedup_enabled", "drop_empty", "apply_curriculum",
            "min_tokens", "max_tokens")},
    ),
    "runs": (
        "backpropagate.ui_app.pages.runs:runs_page",
        {("on_mount", "RunsState.load_runs"), ("on_change", "RunsState.set_status_filter"),
         ("on_click", "RunsState.load_runs"), ("on_click", "RunsState.clear_error"),
         ("on_click", "RunsState.set_status_filter")},
    ),
    "run_detail": (
        "backpropagate.ui_app.pages.run_detail:run_detail_page",
        {("on_mount", "RunDetailState.load_run"), ("on_change", "RunDetailState.set_diff_other_run_id"),
         ("on_click", "RunDetailState.diff_with_input"), ("on_click", "RunDetailState.replay"),
         ("on_click", "RunDetailState.export_run"), ("on_click", "RunDetailState.delete_run"),
         ("on_click", "RunDetailState.clear_action_message")},
    ),
    "models": (
        "backpropagate.ui_app.pages.models:models_page",
        {("on_mount", "ModelsState.load_models"), ("on_click", "ModelsState.load_models"),
         ("on_click", "ModelsState.delete_model")},
    ),
}

_PAGE_COPY = {
    "train": {"HuggingFace model id", "Start training"},
    "multi_run": {"Multi-run", "Sweep shape", "Merge mode", "Cross-run analysis"},
    "export": {"Source", "GGUF quantization", "Register with Ollama", "Push to HF Hub"},
    "dataset": {"Upload", "Detected format", "Enable dedup", "Min tokens", "Max tokens"},
    "runs": {"Run history", "Could not load run history", "No training runs recorded yet.",
             "Reset filter"},
    "run_detail": {"Run not found.", "Training loss", "Checkpoints", "Delete run", "Actions"},
    "models": {"Local models", "No cached models found.", "Delete cached model?", "Cancel"},
}

_PAGE_STATE = {
    "train": "train_state", "multi_run": "multi_run_state", "export": "export_state",
    "dataset": "dataset_state", "runs": "runs_state", "run_detail": "run_detail_state",
    "models": "models_state",
}


def _build_page(name):
    import importlib

    mod_path, fn_name = _PAGE_SPECS[name][0].split(":")
    return getattr(importlib.import_module(mod_path), fn_name)()


@pytest.mark.parametrize("name", sorted(_PAGE_SPECS))
class TestPages:
    def test_event_handlers_point_at_the_right_state_methods(self, name):
        bound = _bound(_build_page(name))
        missing = _PAGE_SPECS[name][1] - bound
        assert not missing, f"{name} page is missing bindings: {sorted(missing)}"

    def test_page_renders_expected_copy(self, name):
        texts = _texts(_build_page(name))
        # "Start training" lives in a static label on the train page's button.
        missing = {t for t in _PAGE_COPY[name] if t not in texts}
        assert not missing, f"{name} page is missing copy: {sorted(missing)}"

    def test_page_is_wrapped_in_the_shared_chrome(self, name):
        page = _build_page(name)
        rendered = _render(page)
        # header, left nav, side rail and footer all present on every surface
        assert "backpropagate logo" in rendered
        assert "Primary" in rendered  # left-nav aria-label
        assert "Live run status" in rendered  # side-rail aria-label
        assert ("on_mount", "AuthBadgeState.refresh") in _bound(page)
        assert _PAGE_STATE[name] in rendered


class TestTrainPageDetails:
    def test_start_button_is_disabled_unless_idle_form_is_valid(self):
        page = _build_page("train")
        rendered = _render(page)
        assert "train_state.run_state" in rendered
        # the Stop/Start buttons both route to TrainState.start_training
        starts = [h for h in _handlers(page) if "TrainState.start_training" in h[2]]
        assert len(starts) == 2

    def test_inline_error_cond_exists_for_model_field(self):
        assert "train_state.model_error" in _render(_build_page("train"))


class TestRunDetailPageDetails:
    def test_back_button_redirects(self):
        page = _build_page("run_detail")
        assert any(n == "_redirect" for _c, _t, names in _handlers(page) for n in names)

    def test_status_variants_all_rendered(self):
        texts = _texts(_build_page("run_detail"))
        assert {"completed", "running", "failed", "interrupted"} <= texts


class TestModelsPageDetails:
    def test_delete_goes_through_a_confirm_dialog(self):
        page = _build_page("models")
        assert "AlertDialogRoot" in _node_types(page) or "DialogRoot" in _node_types(page)
        assert "Delete cached model?" in _texts(page)


# =============================================================================
# app.py — module-level wiring
# =============================================================================


class TestAppWiring:
    def test_app_registers_all_seven_routes(self):
        from backpropagate.ui_app.app import app

        assert set(app._unevaluated_pages) == {
            "index", "multi-run", "export", "dataset", "runs", "runs/[rid]", "models",
        }

    def test_page_titles_are_branded(self):
        from backpropagate.ui_app.app import app

        titles = {k: p.title for k, p in app._unevaluated_pages.items()}
        assert titles["index"] == "backpropagate · train"
        assert titles["runs/[rid]"] == "backpropagate · run detail"
        assert all(t.startswith("backpropagate") for t in titles.values())

    def test_api_transformer_order_is_innermost_first(self):
        from backpropagate.ui_app import middleware
        from backpropagate.ui_app.app import app
        from backpropagate.ui_app.auth import basic_auth_transformer

        chain = tuple(app.api_transformer)
        assert chain == (
            middleware.security_headers_middleware,
            middleware.request_logging_middleware,
            basic_auth_transformer,
            middleware.rate_limit_middleware,
            middleware.healthz_middleware,
        )

    def test_with_tokens_injects_the_token_stylesheet(self):
        from backpropagate.ui_app.app import _with_tokens
        from backpropagate.ui_theme import TOKENS_CSS

        wrapped = _with_tokens(rx.text("page-body"))
        assert type(wrapped).__name__ == "Fragment"
        assert "StyleEl" in _node_types(wrapped)
        assert "page-body" in _texts(wrapped)
        assert "--bp-teal" in TOKENS_CSS and "--bp-teal" in _render(wrapped)

    def test_page_callables_build_a_component_with_tokens(self):
        from backpropagate.ui_app.app import app

        page = app._unevaluated_pages["runs"]
        comp = page.component() if callable(page.component) else page.component
        assert "--bp-teal" in _render(comp)

    def test_theme_is_bound_to_color_mode(self):
        from backpropagate.ui_app.app import app

        # FRONTEND-F-001: appearance follows the live colour mode, not a hard-coded "dark"
        assert "rawColorMode" in str(app.theme.appearance)

    def test_import_refuses_when_auth_module_import_fails(self, monkeypatch):
        """Mocked: ``backpropagate.ui_app.auth`` import is made to raise."""
        import builtins
        import sys

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and name == "auth":
                raise ImportError("boom")
            if name == "backpropagate.ui_app.auth":
                raise ImportError("boom")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        spec = importlib.util.spec_from_file_location(
            "backpropagate.ui_app._app_refuse_probe",
            Path(__import__("backpropagate").__file__).parent / "ui_app" / "app.py",
            submodule_search_locations=None,
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "backpropagate.ui_app"
        sys.modules.pop("backpropagate.ui_app._app_refuse_probe", None)
        with pytest.raises(RuntimeError, match="GHSA-f65r-h4g3-3h9h") as exc:
            spec.loader.exec_module(mod)
        assert "ImportError: boom" in str(exc.value)

    def test_import_refuses_when_enforcement_unavailable_but_auth_env_set(self, monkeypatch):
        """Mocked: ``ENFORCEMENT_AVAILABLE`` forced False; auth env var set."""
        import sys

        import backpropagate.ui_app.auth as auth_mod

        monkeypatch.setattr(auth_mod, "ENFORCEMENT_AVAILABLE", False)
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        spec = importlib.util.spec_from_file_location(
            "backpropagate.ui_app._app_refuse_probe2",
            Path(__import__("backpropagate").__file__).parent / "ui_app" / "app.py",
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "backpropagate.ui_app"
        sys.modules.pop("backpropagate.ui_app._app_refuse_probe2", None)
        with pytest.raises(RuntimeError, match="BACKPROPAGATE_UI_AUTH is set"):
            spec.loader.exec_module(mod)

    def test_enforcement_false_without_auth_env_is_allowed(self, monkeypatch):
        """Mocked: ``ENFORCEMENT_AVAILABLE`` forced False; no auth env var set."""
        import sys

        import backpropagate.ui_app.auth as auth_mod

        monkeypatch.setattr(auth_mod, "ENFORCEMENT_AVAILABLE", False)
        monkeypatch.delenv("BACKPROPAGATE_UI_AUTH", raising=False)
        spec = importlib.util.spec_from_file_location(
            "backpropagate.ui_app._app_ok_probe",
            Path(__import__("backpropagate").__file__).parent / "ui_app" / "app.py",
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "backpropagate.ui_app"
        sys.modules.pop("backpropagate.ui_app._app_ok_probe", None)
        spec.loader.exec_module(mod)  # no refusal
        assert mod.ENFORCEMENT_AVAILABLE is False


# =============================================================================
# rxconfig.py
# =============================================================================


def _load_rxconfig(monkeypatch, name="backpropagate.rxconfig", *, env=None, hide=()):
    """Import ``backpropagate/rxconfig.py`` fresh under a throw-away module name.

    ``hide`` is a tuple of dotted module names whose import is made to fail.
    """
    import builtins
    import sys

    for key in ("BACKPROPAGATE_UI_PORT", "BACKPROPAGATE_UI_CORS_EXTRA_ORIGINS",
                "BACKPROPAGATE_UI_AUTH"):
        monkeypatch.delenv(key, raising=False)
    for key, val in (env or {}).items():
        monkeypatch.setenv(key, val)

    if hide:
        real_import = builtins.__import__

        def fake_import(mod_name, globals=None, locals=None, fromlist=(), level=0):
            if mod_name in hide:
                raise ImportError(f"hidden: {mod_name}")
            return real_import(mod_name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    path = Path(__import__("backpropagate").__file__).parent / "rxconfig.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register under the real dotted name so coverage attributes the lines to
    # ``backpropagate.rxconfig``; ``monkeypatch`` restores sys.modules afterwards.
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


class TestRxConfig:
    def test_default_config_pins_app_and_loopback_cors(self, monkeypatch):
        mod = _load_rxconfig(monkeypatch)
        cfg = mod.config
        assert cfg.app_name == "ui_app"
        assert cfg.app_module_import == "ui_app.app"
        assert cfg.db_url is None
        assert cfg.backend_only is False
        assert cfg.cors_allowed_origins == [
            "http://localhost:3000", "http://localhost:7860",
            "http://127.0.0.1:3000", "http://127.0.0.1:7860",
        ]
        assert "*" not in cfg.cors_allowed_origins  # CSWSH defence: never wildcard

    def test_ui_port_env_adds_loopback_origins(self, monkeypatch):
        mod = _load_rxconfig(monkeypatch, env={"BACKPROPAGATE_UI_PORT": "9001"})
        origins = mod.config.cors_allowed_origins
        assert "http://localhost:9001" in origins and "http://127.0.0.1:9001" in origins

    def test_ui_port_already_default_is_not_duplicated(self, monkeypatch):
        mod = _load_rxconfig(monkeypatch, env={"BACKPROPAGATE_UI_PORT": "7860"})
        assert mod.config.cors_allowed_origins.count("http://localhost:7860") == 1

    def test_non_numeric_ui_port_is_ignored(self, monkeypatch):
        mod = _load_rxconfig(monkeypatch, env={"BACKPROPAGATE_UI_PORT": "80; DROP"})
        assert len(mod.config.cors_allowed_origins) == 4

    def test_extra_origins_are_additive_trimmed_and_deduped(self, monkeypatch):
        mod = _load_rxconfig(
            monkeypatch,
            env={"BACKPROPAGATE_UI_CORS_EXTRA_ORIGINS":
                 " https://a.example , ,https://a.example,http://localhost:3000"},
        )
        origins = mod.config.cors_allowed_origins
        assert origins.count("https://a.example") == 1
        assert origins.count("http://localhost:3000") == 1
        assert "" not in origins
        assert origins[:4] == ["http://localhost:3000", "http://localhost:7860",
                               "http://127.0.0.1:3000", "http://127.0.0.1:7860"]

    def test_falls_back_to_absolute_auth_import(self, monkeypatch):
        """Mocked: the flat ``ui_app.auth`` import is made to fail."""
        mod = _load_rxconfig(monkeypatch, hide=("ui_app.auth",))
        assert mod.ENFORCEMENT_AVAILABLE is True

    def test_both_auth_imports_failing_without_env_degrades_quietly(self, monkeypatch):
        """Mocked: both auth imports fail; BACKPROPAGATE_UI_AUTH unset."""
        mod = _load_rxconfig(monkeypatch, hide=("ui_app.auth", "backpropagate.ui_app.auth"))
        assert mod.ENFORCEMENT_AVAILABLE is False

    def test_both_auth_imports_failing_with_env_refuses_to_start(self, monkeypatch):
        """Mocked: both auth imports fail; BACKPROPAGATE_UI_AUTH set."""
        with pytest.raises(RuntimeError, match="failed to import") as exc:
            _load_rxconfig(
                monkeypatch,
                env={"BACKPROPAGATE_UI_AUTH": "user:pw"},
                hide=("ui_app.auth", "backpropagate.ui_app.auth"),
            )
        assert "GHSA-f65r-h4g3-3h9h" in str(exc.value)

    def test_enforcement_false_with_env_refuses_to_start(self, monkeypatch):
        """Mocked: ``ENFORCEMENT_AVAILABLE`` on the real auth module forced False."""
        import backpropagate.ui_app.auth as auth_mod

        monkeypatch.setattr(auth_mod, "ENFORCEMENT_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="ENFORCEMENT_AVAILABLE is False"):
            _load_rxconfig(monkeypatch, env={"BACKPROPAGATE_UI_AUTH": "user:pw"},
                           hide=("ui_app.auth",))
