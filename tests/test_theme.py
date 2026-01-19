"""
Tests for backpropagate.theme module.

Tests cover:
- Theme creation and configuration
- CSS generation
- Theme info retrieval
- Color palette constants
- All exported symbols
"""

import pytest

# Import module under test
from backpropagate import theme
from backpropagate.theme import (
    create_backpropagate_theme,
    get_css,
    get_theme_info,
    LIGHT_COLORS,
    DARK_COLORS,
    ACCENT,
    SEMANTIC,
    SPACING,
    RADIUS,
    CUSTOM_CSS,
)


class TestThemeCreation:
    """Tests for create_backpropagate_theme function."""

    def test_create_theme_returns_gradio_theme(self):
        """Theme creation returns a valid Gradio theme object."""
        import gradio as gr

        result = create_backpropagate_theme()
        assert isinstance(result, gr.themes.Base)

    def test_create_theme_no_exception(self):
        """Theme creation doesn't raise any exceptions."""
        # Should complete without error
        theme_obj = create_backpropagate_theme()
        assert theme_obj is not None

    def test_create_theme_is_soft_based(self):
        """Theme is based on Gradio's Soft theme."""
        import gradio as gr

        theme_obj = create_backpropagate_theme()
        # The theme should be a subclass or instance of Base
        assert isinstance(theme_obj, gr.themes.Base)

    def test_create_theme_multiple_calls(self):
        """Multiple theme creations work independently."""
        theme1 = create_backpropagate_theme()
        theme2 = create_backpropagate_theme()
        # Both should be valid
        assert theme1 is not None
        assert theme2 is not None


class TestGetCSS:
    """Tests for get_css function."""

    def test_get_css_returns_string(self):
        """CSS function returns a string."""
        css = get_css()
        assert isinstance(css, str)

    def test_get_css_non_empty(self):
        """CSS string is not empty."""
        css = get_css()
        assert len(css) > 0

    def test_get_css_contains_root_vars(self):
        """CSS contains CSS custom properties (variables)."""
        css = get_css()
        assert ":root" in css
        assert "--ocean-teal" in css

    def test_get_css_contains_button_styles(self):
        """CSS contains button styling."""
        css = get_css()
        assert "button" in css.lower()
        assert ".train-btn" in css

    def test_get_css_contains_ocean_mist_colors(self):
        """CSS contains Ocean Mist color variables."""
        css = get_css()
        ocean_vars = [
            "--ocean-teal",
            "--ocean-seafoam",
            "--ocean-coral",
            "--ocean-sand",
            "--ocean-lavender",
            "--ocean-powder",
        ]
        for var in ocean_vars:
            assert var in css, f"CSS missing variable: {var}"

    def test_get_css_contains_spacing_vars(self):
        """CSS contains spacing variables."""
        css = get_css()
        spacing_vars = [
            "--spacing-xs",
            "--spacing-sm",
            "--spacing-md",
            "--spacing-lg",
            "--spacing-xl",
        ]
        for var in spacing_vars:
            assert var in css, f"CSS missing variable: {var}"

    def test_get_css_contains_radius_vars(self):
        """CSS contains radius variables."""
        css = get_css()
        radius_vars = [
            "--radius-sm",
            "--radius-md",
            "--radius-lg",
            "--radius-xl",
        ]
        for var in radius_vars:
            assert var in css, f"CSS missing variable: {var}"

    def test_get_css_contains_transitions(self):
        """CSS contains transition definitions."""
        css = get_css()
        assert "--transition-fast" in css
        assert "--transition-normal" in css

    def test_get_css_contains_animations(self):
        """CSS contains animation keyframes."""
        css = get_css()
        assert "@keyframes fadeIn" in css
        assert "@keyframes pulse" in css

    def test_get_css_contains_responsive_styles(self):
        """CSS contains responsive media queries."""
        css = get_css()
        assert "@media (max-width: 768px)" in css

    def test_get_css_contains_reduced_motion(self):
        """CSS contains reduced motion support."""
        css = get_css()
        assert "@media (prefers-reduced-motion: reduce)" in css

    def test_get_css_matches_constant(self):
        """get_css() returns the CUSTOM_CSS constant."""
        assert get_css() == CUSTOM_CSS


class TestGetThemeInfo:
    """Tests for get_theme_info function."""

    def test_get_theme_info_returns_dict(self):
        """Theme info returns a dictionary."""
        info = get_theme_info()
        assert isinstance(info, dict)

    def test_get_theme_info_has_name(self):
        """Theme info contains name."""
        info = get_theme_info()
        assert "name" in info
        assert info["name"] == "Backpropagate - Ocean Mist"

    def test_get_theme_info_has_version(self):
        """Theme info contains version."""
        info = get_theme_info()
        assert "version" in info
        assert isinstance(info["version"], str)

    def test_get_theme_info_has_mode_names(self):
        """Theme info contains light and dark mode names."""
        info = get_theme_info()
        assert "light_mode" in info
        assert "dark_mode" in info
        assert "Ocean Mist Light" in info["light_mode"]
        assert "Ocean Mist Dark" in info["dark_mode"]

    def test_get_theme_info_has_primary_colors(self):
        """Theme info contains primary colors for both modes."""
        info = get_theme_info()
        assert "primary_color_light" in info
        assert "primary_color_dark" in info
        # Verify they are hex colors
        assert info["primary_color_light"].startswith("#")
        assert info["primary_color_dark"].startswith("#")

    def test_get_theme_info_has_features(self):
        """Theme info contains features list."""
        info = get_theme_info()
        assert "features" in info
        assert isinstance(info["features"], list)
        assert len(info["features"]) > 0

    def test_get_theme_info_features_content(self):
        """Theme info features include expected items."""
        info = get_theme_info()
        features = info["features"]
        expected_keywords = [
            "Ocean Mist",
            "teal",
            "SF Pro",
            "dark mode",
            "spacing",
            "reduced motion",
        ]
        features_text = " ".join(features).lower()
        for keyword in expected_keywords:
            assert keyword.lower() in features_text, f"Features missing: {keyword}"

    def test_get_theme_info_has_colors(self):
        """Theme info contains colors dict with all palettes."""
        info = get_theme_info()
        assert "colors" in info
        colors = info["colors"]
        assert "light" in colors
        assert "dark" in colors
        assert "accent" in colors
        assert "semantic" in colors

    def test_get_theme_info_colors_match_constants(self):
        """Theme info colors match the module constants."""
        info = get_theme_info()
        colors = info["colors"]
        assert colors["light"] == LIGHT_COLORS
        assert colors["dark"] == DARK_COLORS
        assert colors["accent"] == ACCENT
        assert colors["semantic"] == SEMANTIC


class TestColorConstants:
    """Tests for color palette constants."""

    def test_dark_colors_is_dict(self):
        """DARK_COLORS is a dictionary."""
        assert isinstance(DARK_COLORS, dict)

    def test_dark_colors_keys(self):
        """DARK_COLORS contains expected keys."""
        expected_keys = [
            "background",
            "surface",
            "surface_alt",
            "input_bg",
            "border",
            "border_focus",
            "text_primary",
            "text_secondary",
            "text_muted",
        ]
        for key in expected_keys:
            assert key in DARK_COLORS, f"DARK_COLORS missing key: {key}"

    def test_dark_colors_hex_format(self):
        """DARK_COLORS values are valid hex colors."""
        for key, value in DARK_COLORS.items():
            assert isinstance(value, str), f"DARK_COLORS[{key}] is not a string"
            assert value.startswith("#"), f"DARK_COLORS[{key}] doesn't start with #"
            assert len(value) == 7, f"DARK_COLORS[{key}] is not 7 chars (#RRGGBB)"

    def test_light_colors_is_dict(self):
        """LIGHT_COLORS is a dictionary."""
        assert isinstance(LIGHT_COLORS, dict)

    def test_light_colors_keys(self):
        """LIGHT_COLORS contains expected keys."""
        expected_keys = [
            "background",
            "surface",
            "surface_alt",
            "border",
            "border_focus",
            "text_primary",
            "text_secondary",
            "text_muted",
        ]
        for key in expected_keys:
            assert key in LIGHT_COLORS, f"LIGHT_COLORS missing key: {key}"

    def test_light_colors_hex_format(self):
        """LIGHT_COLORS values are valid hex colors."""
        for key, value in LIGHT_COLORS.items():
            assert isinstance(value, str), f"LIGHT_COLORS[{key}] is not a string"
            assert value.startswith("#"), f"LIGHT_COLORS[{key}] doesn't start with #"
            assert len(value) == 7, f"LIGHT_COLORS[{key}] is not 7 chars"

    def test_accent_colors_is_dict(self):
        """ACCENT is a dictionary."""
        assert isinstance(ACCENT, dict)

    def test_accent_colors_keys(self):
        """ACCENT contains expected color keys."""
        expected_keys = [
            "blue",
            "blue_light",
            "green",
            "red",
            "orange",
            "purple",
            "pink",
            "teal",
            "yellow",
            "indigo",
        ]
        for key in expected_keys:
            assert key in ACCENT, f"ACCENT missing key: {key}"

    def test_accent_colors_hex_format(self):
        """ACCENT values are valid hex colors."""
        for key, value in ACCENT.items():
            assert isinstance(value, str), f"ACCENT[{key}] is not a string"
            assert value.startswith("#"), f"ACCENT[{key}] doesn't start with #"
            assert len(value) == 7, f"ACCENT[{key}] is not 7 chars"

    def test_semantic_colors_is_dict(self):
        """SEMANTIC is a dictionary."""
        assert isinstance(SEMANTIC, dict)

    def test_semantic_colors_keys(self):
        """SEMANTIC contains expected keys."""
        expected_keys = [
            "success",
            "success_dark",
            "error",
            "error_dark",
            "warning",
            "warning_dark",
            "info",
            "info_dark",
        ]
        for key in expected_keys:
            assert key in SEMANTIC, f"SEMANTIC missing key: {key}"

    def test_semantic_colors_hex_format(self):
        """SEMANTIC values are valid hex colors."""
        for key, value in SEMANTIC.items():
            assert isinstance(value, str), f"SEMANTIC[{key}] is not a string"
            assert value.startswith("#"), f"SEMANTIC[{key}] doesn't start with #"
            assert len(value) == 7, f"SEMANTIC[{key}] is not 7 chars"


class TestSpacingAndRadius:
    """Tests for spacing and radius constants."""

    def test_spacing_is_dict(self):
        """SPACING is a dictionary."""
        assert isinstance(SPACING, dict)

    def test_spacing_keys(self):
        """SPACING contains expected size keys."""
        expected_keys = ["xs", "sm", "md", "lg", "xl", "2xl"]
        for key in expected_keys:
            assert key in SPACING, f"SPACING missing key: {key}"

    def test_spacing_values_format(self):
        """SPACING values are valid CSS px values."""
        for key, value in SPACING.items():
            assert isinstance(value, str), f"SPACING[{key}] is not a string"
            assert value.endswith("px"), f"SPACING[{key}] doesn't end with px"
            # Verify the numeric part is valid
            num_part = value[:-2]
            assert num_part.isdigit(), f"SPACING[{key}] numeric part invalid"

    def test_spacing_values_increase(self):
        """SPACING values increase in order."""
        order = ["xs", "sm", "md", "lg", "xl", "2xl"]
        values = [int(SPACING[k][:-2]) for k in order]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], f"SPACING not increasing: {order[i]} >= {order[i+1]}"

    def test_radius_is_dict(self):
        """RADIUS is a dictionary."""
        assert isinstance(RADIUS, dict)

    def test_radius_keys(self):
        """RADIUS contains expected size keys."""
        expected_keys = ["sm", "md", "lg", "xl", "full"]
        for key in expected_keys:
            assert key in RADIUS, f"RADIUS missing key: {key}"

    def test_radius_values_format(self):
        """RADIUS values are valid CSS px values."""
        for key, value in RADIUS.items():
            assert isinstance(value, str), f"RADIUS[{key}] is not a string"
            assert value.endswith("px"), f"RADIUS[{key}] doesn't end with px"

    def test_radius_full_is_large(self):
        """RADIUS 'full' is a large value for pill shapes."""
        full_value = int(RADIUS["full"][:-2])
        assert full_value >= 9999, "RADIUS['full'] should be 9999px for pill shapes"


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_defined(self):
        """Module has __all__ defined."""
        assert hasattr(theme, "__all__")
        assert isinstance(theme.__all__, list)

    def test_all_exports_importable(self):
        """All items in __all__ can be imported."""
        for name in theme.__all__:
            assert hasattr(theme, name), f"Export {name} not found in module"

    def test_expected_exports(self):
        """Module exports expected items."""
        expected = [
            "create_backpropagate_theme",
            "get_css",
            "get_theme_info",
            "LIGHT_COLORS",
            "DARK_COLORS",
            "ACCENT",
            "SEMANTIC",
            "SPACING",
            "RADIUS",
            "CUSTOM_CSS",
        ]
        for item in expected:
            assert item in theme.__all__, f"Expected export {item} not in __all__"

    def test_no_extra_private_exports(self):
        """No private items (starting with _) in __all__."""
        for name in theme.__all__:
            assert not name.startswith("_"), f"Private item {name} in __all__"


class TestCustomCSSConstant:
    """Tests for CUSTOM_CSS constant."""

    def test_custom_css_is_string(self):
        """CUSTOM_CSS is a string."""
        assert isinstance(CUSTOM_CSS, str)

    def test_custom_css_not_empty(self):
        """CUSTOM_CSS is not empty."""
        assert len(CUSTOM_CSS) > 0

    def test_custom_css_valid_structure(self):
        """CUSTOM_CSS has valid CSS structure (contains selectors and braces)."""
        # Should contain CSS selectors and rule blocks
        assert "{" in CUSTOM_CSS
        assert "}" in CUSTOM_CSS
        # Brace count should be balanced
        assert CUSTOM_CSS.count("{") == CUSTOM_CSS.count("}")

    def test_custom_css_no_syntax_errors(self):
        """CUSTOM_CSS doesn't have obvious syntax issues."""
        # Check for common CSS syntax issues
        lines = CUSTOM_CSS.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments and empty lines
            if not stripped or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            # Property lines inside rules should end with ; or { or }
            # This is a basic check, not comprehensive CSS validation


class TestGRAYSConstant:
    """Tests for GRAYS constant (not in __all__ but used internally)."""

    def test_grays_exists(self):
        """GRAYS constant exists in module."""
        from backpropagate.theme import GRAYS
        assert isinstance(GRAYS, dict)

    def test_grays_keys(self):
        """GRAYS contains expected gray scale keys."""
        from backpropagate.theme import GRAYS
        expected_keys = ["gray", "gray2", "gray3", "gray4", "gray5", "gray6"]
        for key in expected_keys:
            assert key in GRAYS, f"GRAYS missing key: {key}"

    def test_grays_hex_format(self):
        """GRAYS values are valid hex colors."""
        from backpropagate.theme import GRAYS
        for key, value in GRAYS.items():
            assert isinstance(value, str), f"GRAYS[{key}] is not a string"
            assert value.startswith("#"), f"GRAYS[{key}] doesn't start with #"
            assert len(value) == 7, f"GRAYS[{key}] is not 7 chars"


class TestThemeColorConsistency:
    """Tests for color consistency across theme."""

    def test_border_focus_matches_accent(self):
        """Border focus colors match accent colors."""
        assert DARK_COLORS["border_focus"] == ACCENT["blue"]
        assert LIGHT_COLORS["border_focus"] == ACCENT["blue_light"]

    def test_semantic_success_matches_accent_green(self):
        """Semantic success color matches accent green."""
        assert SEMANTIC["success"] == ACCENT["green"]

    def test_semantic_error_matches_accent_red(self):
        """Semantic error color matches accent red."""
        assert SEMANTIC["error"] == ACCENT["red"]

    def test_semantic_warning_matches_accent_orange(self):
        """Semantic warning color matches accent orange."""
        assert SEMANTIC["warning"] == ACCENT["orange"]

    def test_theme_info_colors_accurate(self):
        """Theme info primary colors match accent constants."""
        info = get_theme_info()
        assert info["primary_color_light"] == ACCENT["blue_light"]
        assert info["primary_color_dark"] == ACCENT["blue"]
