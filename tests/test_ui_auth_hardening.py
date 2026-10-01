"""Regression tests for the ``backprop ui`` auth hardening.

Two defects, both reachable from the CLI:

1. **The default launch was unauthenticated.** ``backprop ui`` (no ``--auth``)
   never set ``BACKPROPAGATE_UI_LAUNCH_TOKEN``, so the middleware fell through
   to ``NO_AUTH_LOCAL_ONLY`` while the handbook promised a per-launch token +
   0600 lock file. Any local process, or any web page that could reach
   localhost, could drive the UI.
2. **Passwords were stored in plaintext.** ``--auth user:pass`` put the
   plaintext in the child's ``BACKPROPAGATE_UI_AUTH`` env var and in the lock
   file, and the session-cookie HMAC key was the unsalted ``SHA-256(user:pass)``.

These tests fail on the pre-fix code: they assert on the env handed to the
Reflex child, the lock file, the banner, and they drive the real ASGI
middleware (no mocks of the auth logic).

The module deliberately imports nothing that only exists after the fix at the
top level, so it fails (rather than errors at collection) on the old code.
"""

from __future__ import annotations

import base64
import hashlib
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

httpx = pytest.importorskip("httpx", reason="httpx>=0.27 is required for ASGI middleware tests.")

from backpropagate import cli  # noqa: E402
from tests.helpers.asgi import basic_auth_header, make_asgi_client, stub_asgi_http_app  # noqa: E402
from tests.helpers.cli_cov_support import parse  # noqa: E402
from tests.helpers.ws import WSMessageRecorder, make_ws_scope  # noqa: E402

_AUTH_VARS = (
    "BACKPROPAGATE_UI_AUTH",
    "BACKPROPAGATE_UI_AUTH_USER",
    "BACKPROPAGATE_UI_AUTH_VERIFIER",
    "BACKPROPAGATE_UI_LAUNCH_TOKEN",
    "BACKPROPAGATE_UI_SHARE_HOST",
    "BACKPROPAGATE_UI_HOST_BIND",
    "BACKPROPAGATE_UI_QUIET",
)

PASSWORD = "Tr0ub4dor&3-correct-horse"
HOST = "http://127.0.0.1:7862"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture
def launch(tmp_path, monkeypatch, capsys):
    """Run the real ``cmd_ui`` against a fake Reflex child.

    Returns a callable ``launch(argv) -> Launch`` where ``Launch`` carries the
    env the child received, the lock files visible WHILE the child ran
    (name -> (text, mode)), everything printed, and the lock directory.
    """
    for name in _AUTH_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg"))
    monkeypatch.setattr(cli, "_find_port_in_use", lambda host, ports: None)
    lock_dir = tmp_path / "xdg" / "backpropagate"

    def run(argv: list[str]) -> SimpleNamespace:
        seen: dict = {}

        def fake_run(cmd, env=None, cwd=None):
            seen["env"] = dict(env or {})
            seen["locks"] = {
                p.name: (p.read_text(encoding="utf-8"), stat.S_IMODE(p.stat().st_mode))
                for p in (sorted(lock_dir.glob("*")) if lock_dir.exists() else [])
            }
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cli, "_run_reflex", fake_run)
        assert cli.cmd_ui(parse(["ui", *argv])) == cli.EXIT_OK
        out = capsys.readouterr()
        return SimpleNamespace(
            env=seen["env"],
            locks=seen["locks"],
            stdout=out.out,
            stderr=out.err,
            lock_dir=lock_dir,
            leftover=sorted(lock_dir.glob("*")) if lock_dir.exists() else [],
        )

    return run


def _apply_child_env(monkeypatch, env: dict[str, str]) -> None:
    """Make the in-process middleware see exactly the env the child got."""
    for name in _AUTH_VARS:
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        if key.startswith("BACKPROPAGATE_UI_"):
            monkeypatch.setenv(key, value)


def _client(base_url: str = HOST) -> httpx.AsyncClient:
    from backpropagate.ui_app.auth import basic_auth_transformer

    return make_asgi_client(basic_auth_transformer(stub_asgi_http_app), base_url=base_url)


def _session_cookie(response: httpx.Response) -> str:
    set_cookie = response.headers["set-cookie"]
    assert "HttpOnly" in set_cookie
    return set_cookie.split(";", 1)[0]  # "backprop_sess=<value>"


# ---------------------------------------------------------------------------
# Defect 1: the default launch must be token-authenticated
# ---------------------------------------------------------------------------


class TestDefaultLaunchIsTokenAuthenticated:
    def test_child_env_carries_a_random_launch_token(self, launch):
        first = launch([]).env
        second = launch([]).env
        token = first.get("BACKPROPAGATE_UI_LAUNCH_TOKEN", "")
        assert len(token) >= 40  # secrets.token_urlsafe(32) is 43 chars
        assert set(token) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        assert second["BACKPROPAGATE_UI_LAUNCH_TOKEN"] != token  # per launch
        # No credential variables in token mode.
        assert "BACKPROPAGATE_UI_AUTH" not in first
        assert "BACKPROPAGATE_UI_AUTH_VERIFIER" not in first

    def test_banner_prints_the_full_url_with_the_token(self, launch):
        result = launch([])
        token = result.env["BACKPROPAGATE_UI_LAUNCH_TOKEN"]
        assert f"http://127.0.0.1:7862/?token={token}" in result.stderr
        assert "token (auto-generated)" in result.stderr

    def test_lock_file_holds_the_token_while_running_and_is_deleted_on_exit(self, launch):
        result = launch([])
        token = result.env["BACKPROPAGATE_UI_LAUNCH_TOKEN"]
        assert list(result.locks) == ["session-7862.lock"]
        assert result.locks["session-7862.lock"][0] == token
        assert result.leftover == []  # removed on shutdown, as documented

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
    def test_lock_file_mode_is_0600_on_posix(self, launch):
        mode = launch([]).locks["session-7862.lock"][1]
        assert mode == 0o600

    @pytest.mark.skipif(os.name == "posix", reason="Windows-specific protection note")
    def test_windows_says_what_protects_the_lock_file(self, launch):
        out = launch([]).stdout
        assert "lock-file" in out and "ACL" in out

    def test_ambient_credentials_cannot_downgrade_the_default_launch(self, launch, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", "ambient-token")
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "ambient:creds")
        env = launch([]).env
        assert env["BACKPROPAGATE_UI_LAUNCH_TOKEN"] != "ambient-token"
        assert "BACKPROPAGATE_UI_AUTH" not in env

    async def test_middleware_rejects_without_token_and_accepts_with_it(self, launch, monkeypatch):
        env = launch([]).env
        token = env["BACKPROPAGATE_UI_LAUNCH_TOKEN"]
        _apply_child_env(monkeypatch, env)

        async with _client() as client:
            # The old default (NO_AUTH_LOCAL_ONLY) answered 200 here.
            assert (await client.get("/")).status_code == 401
            assert (await client.get("/", params={"token": "wrong-" + token})).status_code == 401
            assert (await client.get("/", params={"token": ""})).status_code == 401

            ok = await client.get("/", params={"token": token})
            assert ok.status_code == 302  # cookie set, token stripped from the URL
            assert ok.headers["location"] == "/"
            cookie = _session_cookie(ok)

            # The cookie alone now opens the app.
            assert (await client.get("/", headers={"cookie": cookie})).status_code == 200

    async def test_websocket_needs_the_session_cookie(self, launch, monkeypatch):
        env = launch([]).env
        token = env["BACKPROPAGATE_UI_LAUNCH_TOKEN"]
        _apply_child_env(monkeypatch, env)

        async def receive():
            return {"type": "websocket.connect"}

        from backpropagate.ui_app.auth import basic_auth_transformer

        app = basic_auth_transformer(stub_asgi_http_app)

        rejected = WSMessageRecorder()
        await app(make_ws_scope(host="127.0.0.1:7862", origin="http://127.0.0.1:7862"), receive, rejected)
        assert rejected.close_code == 4401
        assert not rejected.accepted_before_closed

        async with _client() as client:
            cookie = _session_cookie(await client.get("/", params={"token": token}))
        name, value = cookie.split("=", 1)
        accepted = WSMessageRecorder()
        await app(
            make_ws_scope(
                host="127.0.0.1:7862", origin="http://127.0.0.1:7862", cookies={name: value}
            ),
            receive,
            accepted,
        )
        assert accepted.accepted

    async def test_host_header_is_validated_in_token_mode(self, launch, monkeypatch):
        """DNS-rebinding defense: a foreign Host is refused before any auth check."""
        env = launch([]).env
        token = env["BACKPROPAGATE_UI_LAUNCH_TOKEN"]
        _apply_child_env(monkeypatch, env)
        async with _client("http://evil.example:7862") as client:
            response = await client.get("/", params={"token": token})
        assert response.status_code == 421


# ---------------------------------------------------------------------------
# Defect 2: --auth must not put the plaintext anywhere
# ---------------------------------------------------------------------------


class TestExplicitCredentialsNeverPlaintext:
    def _argv(self, tmp_path, via_file: bool) -> list[str]:
        if not via_file:
            return ["--auth", f"alice:{PASSWORD}"]
        creds = tmp_path / "ui.auth"
        creds.write_text(f"alice:{PASSWORD}", encoding="utf-8")
        return ["--auth-file", str(creds)]

    @pytest.mark.parametrize("via_file", [False, True], ids=["--auth", "--auth-file"])
    def test_no_plaintext_in_child_env_lock_dir_or_output(self, launch, tmp_path, via_file):
        result = launch(self._argv(tmp_path, via_file))

        assert "BACKPROPAGATE_UI_AUTH" not in result.env
        for key, value in result.env.items():
            if key.startswith("BACKPROPAGATE_"):
                assert PASSWORD not in value, f"{key} carries the plaintext password"
        # Nothing credential-shaped on disk: no lock file at all for --auth.
        assert result.locks == {}
        assert result.leftover == []
        for text, _mode in result.locks.values():
            assert PASSWORD not in text
        assert PASSWORD not in result.stdout
        assert PASSWORD not in result.stderr
        # And no launch token either: explicit credentials replace it.
        assert "BACKPROPAGATE_UI_LAUNCH_TOKEN" not in result.env

    def test_child_gets_user_plus_scrypt_verifier(self, launch, tmp_path):
        env = launch(self._argv(tmp_path, False)).env
        assert env["BACKPROPAGATE_UI_AUTH_USER"] == "alice"
        parts = env["BACKPROPAGATE_UI_AUTH_VERIFIER"].split("$")
        assert parts[0] == "scrypt"
        assert (int(parts[1]), int(parts[2]), int(parts[3])) == (2**14, 8, 1)
        assert len(base64.b64decode(parts[4])) == 16  # 16-byte random salt
        assert len(base64.b64decode(parts[5])) == 32

    def test_each_launch_uses_a_fresh_salt(self, launch, tmp_path):
        a = launch(self._argv(tmp_path, False)).env["BACKPROPAGATE_UI_AUTH_VERIFIER"]
        b = launch(self._argv(tmp_path, False)).env["BACKPROPAGATE_UI_AUTH_VERIFIER"]
        assert a != b

    async def test_right_password_passes_wrong_one_fails_through_the_middleware(
        self, launch, tmp_path, monkeypatch
    ):
        env = launch(self._argv(tmp_path, False)).env
        _apply_child_env(monkeypatch, env)

        async with _client() as client:
            assert (await client.get("/")).status_code == 401
            right = await client.get("/", headers=basic_auth_header("alice", PASSWORD))
            assert right.status_code == 200
            assert "backprop_sess=" in right.headers["set-cookie"]
            client.cookies.clear()  # the jar now holds a valid session; test credentials, not the cookie
            for user, password in (
                ("alice", PASSWORD + "x"),
                ("alice", PASSWORD.upper()),
                ("alice", ""),
                ("mallory", PASSWORD),
            ):
                bad = await client.get("/", headers=basic_auth_header(user, password))
                assert bad.status_code == 401, (user, password)

    async def test_cookie_key_is_not_derived_from_the_password(self, launch, tmp_path, monkeypatch):
        """The old key was SHA-256('user:pass'); a cookie forged with it must now be refused."""
        from backpropagate.ui_app import auth

        env = launch(self._argv(tmp_path, False)).env
        _apply_child_env(monkeypatch, env)

        key = auth._derive_secret(dict(os.environ))
        for guess in (
            f"alice:{PASSWORD}",
            PASSWORD,
            "alice",
        ):
            assert key != hashlib.sha256(guess.encode()).digest()
        assert len(key) == 32

        forged = auth._sign_cookie("alice", hashlib.sha256(f"alice:{PASSWORD}".encode()).digest())
        async with _client() as client:
            response = await client.get("/", headers={"cookie": f"backprop_sess={forged}"})
        assert response.status_code == 401

        # A cookie minted by the real flow still works.
        async with _client() as client:
            ok = await client.get("/", headers=basic_auth_header("alice", PASSWORD))
            cookie = _session_cookie(ok)
            assert (await client.get("/", headers={"cookie": cookie})).status_code == 200

    def test_cookie_key_is_random_per_process(self):
        code = "from backpropagate.ui_app import auth; print(auth._PROCESS_LOCAL_SECRET.hex())"
        keys = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
            ).stdout.strip()
            for _ in range(2)
        }
        assert len(keys) == 2 and all(len(k) == 64 for k in keys)


# ---------------------------------------------------------------------------
# Direct-Reflex users who set the legacy plaintext env var themselves
# ---------------------------------------------------------------------------


class TestLegacyPlaintextEnvStillAccepted:
    async def test_accepted_and_verified_in_memory(self, monkeypatch):
        from backpropagate.ui_app import auth

        for name in _AUTH_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", f"bob:{PASSWORD}")

        user, verifier = auth._resolve_credential(dict(os.environ))
        assert user == "bob"
        assert verifier.startswith("scrypt$") and PASSWORD not in verifier

        async with _client() as client:
            assert (
                await client.get("/", headers=basic_auth_header("bob", PASSWORD))
            ).status_code == 200
            client.cookies.clear()  # the jar now holds a valid session; test credentials, not the cookie
            assert (
                await client.get("/", headers=basic_auth_header("bob", "nope"))
            ).status_code == 401
            assert (await client.get("/")).status_code == 401

    def test_derivation_is_cached_per_value_not_per_request(self, monkeypatch):
        from backpropagate.ui_app import auth

        env = {"BACKPROPAGATE_UI_AUTH": "carol:pw-for-cache-test"}
        assert auth._resolve_credential(env) == auth._resolve_credential(env)


class TestVerifierEnvHandling:
    async def test_malformed_verifier_fails_closed(self, monkeypatch):
        """A bad verifier must never fall back to open access."""
        from backpropagate.ui_app import auth

        for name in _AUTH_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH_USER", "alice")
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH_VERIFIER", "scrypt$not$a$verifier")
        assert auth._detect_mode() is auth.AuthMode.EXPLICIT_CREDS
        async with _client() as client:
            assert (
                await client.get("/", headers=basic_auth_header("alice", "anything"))
            ).status_code == 401

    async def test_verifier_without_user_fails_closed(self, monkeypatch):
        from backpropagate.ui_security import hash_password

        for name in _AUTH_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH_VERIFIER", hash_password("pw-x"))
        async with _client() as client:
            assert (await client.get("/", headers=basic_auth_header("", "pw-x"))).status_code == 401

    def test_verifier_wins_over_plaintext(self, monkeypatch):
        from backpropagate.ui_app import auth
        from backpropagate.ui_security import hash_password

        verifier = hash_password("the-real-one")
        env = {
            "BACKPROPAGATE_UI_AUTH": "alice:other",
            "BACKPROPAGATE_UI_AUTH_USER": "alice",
            "BACKPROPAGATE_UI_AUTH_VERIFIER": verifier,
        }
        assert auth._resolve_credential(env) == ("alice", verifier)

    def test_import_guards_treat_every_credential_var_as_auth(self, monkeypatch):
        """If the middleware can't load, any auth env var must still refuse to start."""
        from backpropagate import rxconfig

        for name in _AUTH_VARS:
            monkeypatch.delenv(name, raising=False)
        assert rxconfig._any_ui_auth_env() is False
        for name in (
            "BACKPROPAGATE_UI_AUTH",
            "BACKPROPAGATE_UI_AUTH_VERIFIER",
            "BACKPROPAGATE_UI_LAUNCH_TOKEN",
        ):
            monkeypatch.setenv(name, "x")
            assert rxconfig._any_ui_auth_env() is True
            monkeypatch.delenv(name)

    def test_badge_reports_basic_without_leaking_the_verifier(self):
        from backpropagate.ui_security import get_auth_badge_context, hash_password

        verifier = hash_password("pw-badge")
        ctx = get_auth_badge_context(
            {
                "BACKPROPAGATE_UI_AUTH_USER": "alice",
                "BACKPROPAGATE_UI_AUTH_VERIFIER": verifier,
                "BACKPROPAGATE_UI_HOST_BIND": "127.0.0.1",
            }
        )
        assert ctx.mode_key == "basic_local"
        assert ctx.auth_user == "alice"
        assert verifier not in ctx.hover_text and "scrypt" not in ctx.hover_text


# ---------------------------------------------------------------------------
# The verifier primitive
# ---------------------------------------------------------------------------


class TestScryptVerifier:
    def test_round_trip_and_format(self):
        from backpropagate.ui_security import hash_password, verify_password

        verifier = hash_password(PASSWORD)
        kind, n, r, p, salt, digest = verifier.split("$")
        assert kind == "scrypt" and (n, r, p) == (str(2**14), "8", "1")
        assert len(base64.b64decode(salt)) == 16 and len(base64.b64decode(digest)) == 32
        assert verify_password(PASSWORD, verifier)
        assert not verify_password(PASSWORD + " ", verifier)
        assert not verify_password("", verifier)

    def test_salted_so_equal_passwords_differ(self):
        from backpropagate.ui_security import hash_password, verify_password

        a, b = hash_password(PASSWORD), hash_password(PASSWORD)
        assert a != b
        assert verify_password(PASSWORD, a) and verify_password(PASSWORD, b)

    def test_unicode_password(self):
        from backpropagate.ui_security import hash_password, verify_password

        verifier = hash_password("pässwörd-密码")
        assert verify_password("pässwörd-密码", verifier)
        assert not verify_password("passwort-密码", verifier)

    def test_cost_parameters_travel_with_the_verifier(self):
        from backpropagate.ui_security import hash_password, verify_password

        cheap = hash_password("pw", n=2**10, r=8, p=1)
        assert cheap.split("$")[1] == str(2**10)
        assert verify_password("pw", cheap)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "plain-text-password",
            "scrypt$16384$8$1$onlyfour",
            "bcrypt$16384$8$1$AAAA$AAAA",
            "scrypt$notint$8$1$AAAA$AAAA",
            "scrypt$16384$8$1$!!!!$AAAA",
            "scrypt$16383$8$1$AAAA$AAAA",  # n not a power of two
            "scrypt$1$8$1$AAAA$AAAA",
            f"scrypt${2**30}$8$1$AAAA$AAAA",  # absurd cost: refused, not allocated
            "scrypt$16384$0$1$AAAA$AAAA",
            "scrypt$16384$8$1$$AAAA",
        ],
    )
    def test_malformed_verifiers_fail_closed(self, bad):
        from backpropagate.ui_security import is_valid_verifier, verify_password

        assert is_valid_verifier(bad) is False
        assert verify_password("anything", bad) is False

    def test_comparison_is_constant_time_compare_digest(self, monkeypatch):
        import backpropagate.ui_security as sec

        calls = []
        real = sec.hmac.compare_digest
        monkeypatch.setattr(
            sec.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b)
        )
        verifier = sec.hash_password("pw")
        assert sec.verify_password("pw", verifier)
        assert calls
