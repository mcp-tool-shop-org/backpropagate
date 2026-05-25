"""
Legacy-alias tests for the v1.4 ``config`` symbol rename
========================================================

Wave 6a foundation (Wave 5 Decision 3) renamed ``TRAINING_PRESETS`` →
``MULTI_RUN_PRESETS`` in ``backpropagate.config`` to disambiguate from the
v1.3-era ``LORA_PRESETS`` (LoRA-shape preset; CLI ``--lora-preset``).
Both formerly shared the keys ``"fast"`` + ``"quality"`` with semantically
different values, surfaced as a Wave 5 audit operator-trap finding.

The legacy ``TRAINING_PRESETS`` name continues to resolve via a
module-level ``__getattr__`` shim and emits a ``DeprecationWarning``
pointing at the canonical replacement. ``LORA_PRESETS`` was NOT touched —
it's the source of the user-facing ``--lora-preset`` flag values.

The contract:

1. Both the legacy AND canonical names import + resolve.
2. Accessing the LEGACY ``TRAINING_PRESETS`` from
   ``backpropagate.config`` emits a ``DeprecationWarning`` whose message
   pins the deprecation cycle text.
3. The legacy name resolves to the SAME dict object as the canonical name
   (``is`` comparison, not just ``==``).
4. Accessing an attribute that's NOT in the legacy table raises
   ``AttributeError`` per the PEP 562 ``__getattr__`` contract.

The deprecation cycle is locked at advisor 2026-05-25 Q4:

* v1.4 — ``DeprecationWarning``
* v1.5 — ``UserWarning``
* v1.6 — ``AttributeError``
"""

from __future__ import annotations

import re
import warnings

import pytest

# Regex pinning the three load-bearing phases of the deprecation cycle in
# the warning message. Same shape as ``test_ui_security_legacy_aliases``
# so future drift in either site fails consistently.
_DEPRECATION_MESSAGE_REGEX = re.compile(
    r"deprecated in v1\.4.*v1\.5 escalates.*v1\.6 removes",
    re.DOTALL,
)


class TestConfigLegacyAliases:
    """Module-level ``__getattr__`` legacy-alias shim contract."""

    def test_training_presets_emits_deprecation_warning(self):
        """``TRAINING_PRESETS`` emits ``DeprecationWarning`` on access."""
        import backpropagate.config as config

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy_table = config.TRAINING_PRESETS

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1, (
            "Accessing legacy 'TRAINING_PRESETS' must emit DeprecationWarning"
        )
        msg = str(deprecation_warnings[0].message)
        assert "TRAINING_PRESETS" in msg, "Warning must name the legacy symbol"
        assert "MULTI_RUN_PRESETS" in msg, (
            "Warning must name the canonical replacement"
        )
        assert _DEPRECATION_MESSAGE_REGEX.search(msg), (
            f"Warning must pin the v1.4 → v1.5 → v1.6 cycle. Got: {msg!r}"
        )
        # Identity-equal with the canonical: same dict object.
        assert legacy_table is config.MULTI_RUN_PRESETS, (
            "Legacy alias must resolve to the IDENTITY-SAME dict object "
            "as the canonical name."
        )

    def test_training_presets_warning_mentions_lora_presets(self):
        """Warning should disambiguate from ``LORA_PRESETS``.

        The whole REASON for the rename is the namespace collision — the
        warning is the operator's first contact point and should mention
        that ``LORA_PRESETS`` is the OTHER preset table they might want.
        """
        import backpropagate.config as config

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = config.TRAINING_PRESETS

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        msg = str(deprecation_warnings[0].message)
        assert "LORA_PRESETS" in msg, (
            "Warning must mention LORA_PRESETS so operators who reached "
            "for 'TRAINING_PRESETS' get pointed at the OTHER preset table "
            "they might actually have wanted (--lora-preset)."
        )

    def test_multi_run_presets_canonical_does_not_warn(self):
        """Canonical ``MULTI_RUN_PRESETS`` resolves silently."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from backpropagate.config import MULTI_RUN_PRESETS

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0, (
            f"Canonical 'MULTI_RUN_PRESETS' must NOT emit DeprecationWarning. "
            f"Got: {[str(w.message) for w in deprecation_warnings]!r}"
        )
        # Sanity — the canonical dict has the documented keys.
        assert "fast" in MULTI_RUN_PRESETS
        assert "balanced" in MULTI_RUN_PRESETS
        assert "quality" in MULTI_RUN_PRESETS

    def test_legacy_equals_canonical(self):
        """``TRAINING_PRESETS == MULTI_RUN_PRESETS`` and is the same object."""
        import backpropagate.config as config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence the legacy warning
            legacy = config.TRAINING_PRESETS

        # Identity-equal: PEP 562 __getattr__ returns the canonical
        # via ``globals()[new_name]`` — same dict object.
        assert legacy is config.MULTI_RUN_PRESETS

        # And equal as dicts (belt + suspenders for the test reader who's
        # not familiar with PEP 562 semantics).
        assert legacy == config.MULTI_RUN_PRESETS

    def test_lora_presets_untouched(self):
        """``LORA_PRESETS`` was not renamed — verify it's still canonical.

        The rename targets ``TRAINING_PRESETS`` ONLY. ``LORA_PRESETS`` is
        the v1.3 BACKEND-1 user-facing ``--lora-preset`` flag's source and
        must stay accessible without a warning.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from backpropagate.config import LORA_PRESETS

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0, (
            f"LORA_PRESETS must NOT emit DeprecationWarning. Got: "
            f"{[str(w.message) for w in deprecation_warnings]!r}"
        )
        assert "fast" in LORA_PRESETS
        assert "quality" in LORA_PRESETS
        # And the load-bearing distinction: LORA_PRESETS["quality"].r is
        # the rank-256 architecture preset; if these tests pass while
        # quality.r is somehow back to 32 it means we collapsed namespaces.
        assert LORA_PRESETS["quality"].r == 256, (
            "LORA_PRESETS['quality'] must remain the rank-256 v1.3 "
            "BACKEND-1 default (NOT the rank-32 MULTI_RUN_PRESETS shape)."
        )

    def test_get_preset_routes_through_canonical(self):
        """``get_preset`` reads from the canonical ``MULTI_RUN_PRESETS``.

        The function was migrated as part of the rename — verify it points
        at the new dict so ``MULTI_RUN_PRESETS`` is the single source of
        truth and ``TRAINING_PRESETS`` is purely a shim.
        """
        from backpropagate.config import MULTI_RUN_PRESETS, get_preset

        preset = get_preset("balanced")
        assert preset is MULTI_RUN_PRESETS["balanced"], (
            "get_preset must return the IDENTITY-SAME TrainingPreset as "
            "the canonical MULTI_RUN_PRESETS lookup."
        )

    def test_unknown_attribute_raises_attribute_error(self):
        """``__getattr__`` raises ``AttributeError`` for non-legacy names.

        Required by the PEP 562 contract so ``hasattr`` / ``getattr`` with
        a default still work naturally on the module.
        """
        import backpropagate.config as config

        with pytest.raises(AttributeError) as exc_info:
            config.this_symbol_does_not_exist  # noqa: B018
        assert "config" in str(exc_info.value)
        assert "this_symbol_does_not_exist" in str(exc_info.value)

    def test_package_level_alias_is_silent(self):
        """``from backpropagate import TRAINING_PRESETS`` must NOT warn.

        The deprecation surface is the ``backpropagate.config`` module
        (where the rename happened). The top-level ``backpropagate``
        package keeps ``TRAINING_PRESETS`` as a silent back-compat alias
        for ``MULTI_RUN_PRESETS`` so stable public-API code stays
        warning-free until v1.6 (where both names get pruned together).
        """
        # Re-importing inside the warning-catch context — but `backpropagate`
        # is already loaded by pytest, so we have to fetch the attribute
        # rather than import it. Either way, attribute access on a stable
        # module attribute does not trigger the config.py __getattr__.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import backpropagate
            _ = backpropagate.TRAINING_PRESETS

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0, (
            f"Package-level 'backpropagate.TRAINING_PRESETS' alias must "
            f"stay silent. Got: "
            f"{[str(w.message) for w in deprecation_warnings]!r}"
        )
