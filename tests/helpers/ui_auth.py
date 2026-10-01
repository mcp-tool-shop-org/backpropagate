"""Shared assertions for the ``backprop ui`` auth hand-off (child-process env).

``cmd_ui`` never gives the Reflex child the plaintext password. For ``--auth``
it passes ``BACKPROPAGATE_UI_AUTH_USER`` + ``BACKPROPAGATE_UI_AUTH_VERIFIER``
(a salted scrypt verifier); without ``--auth`` it passes a per-launch
``BACKPROPAGATE_UI_LAUNCH_TOKEN``. These helpers let the many tests that
inspect the child env say so in one line.
"""

from __future__ import annotations

from collections.abc import Mapping

from backpropagate.ui_security import verify_password


def assert_child_env_has_verifier(env: Mapping[str, str], user: str, password: str) -> None:
    """The child env authenticates ``user``/``password`` without carrying it in clear."""
    assert "BACKPROPAGATE_UI_AUTH" not in env, "plaintext credential crossed the process boundary"
    assert "BACKPROPAGATE_UI_LAUNCH_TOKEN" not in env
    assert env.get("BACKPROPAGATE_UI_AUTH_USER") == user
    verifier = env.get("BACKPROPAGATE_UI_AUTH_VERIFIER", "")
    assert verifier.startswith("scrypt$")
    assert verify_password(password, verifier)
    assert not verify_password(password + "x", verifier)
    # No env value may contain the password (a verifier is base64, never the password
    # itself). Skipped for very short passwords, which collide with ordinary values.
    if len(password) >= 6:
        for key, value in env.items():
            if key.startswith("BACKPROPAGATE_"):
                assert password not in value, f"{key} contains the plaintext password"
