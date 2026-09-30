"""Patch targets for trl's preference classes, wherever the installed trl keeps them.

trl exported ``ORPOConfig`` / ``ORPOTrainer`` and ``CPOConfig`` / ``CPOTrainer``
at the top level through 0.28 and moved them to ``trl.experimental.orpo`` /
``trl.experimental.cpo`` in 0.29. ``KTOConfig`` / ``KTOTrainer`` are still
top-level in 1.13. ``backpropagate.trainer`` resolves each class the same way:
the top-level name first, then the experimental module.

A test that patches ``"trl.ORPOTrainer"`` therefore patches a name trl >= 0.29
does not have, and ``mock.patch`` raises ``AttributeError`` before the test body
runs. ``trl_patch_target`` returns the dotted path the code under test will
actually read, so one test patches the right object on trl 0.24 and on 1.x.
"""

from __future__ import annotations

_EXPERIMENTAL_MODULE = {
    "ORPOConfig": "trl.experimental.orpo",
    "ORPOTrainer": "trl.experimental.orpo",
    "CPOConfig": "trl.experimental.cpo",
    "CPOTrainer": "trl.experimental.cpo",
    "KTOConfig": "trl.experimental.kto",
    "KTOTrainer": "trl.experimental.kto",
}


def trl_patch_target(name: str) -> str:
    """Dotted path of trl class ``name`` where ``backpropagate.trainer`` finds it."""
    import trl

    if hasattr(trl, name):
        return f"trl.{name}"
    # No import needed here: mock.patch imports the target module itself.
    return f"{_EXPERIMENTAL_MODULE[name]}.{name}"
