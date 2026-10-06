"""Warm the Reflex frontend-install cache marker without touching the network.

Runs as ``python -m backpropagate.ui_marker_warmup <port> <host>`` from the UI
working directory, as a child step of ``cmd_ui`` (see ui_frontend.py). Reflex
0.9.x decides whether to run ``bun add`` by comparing a fingerprint stored in
``.web/reflex.install_frontend_packages.cached`` (a ``pickle`` of
``(payload, value)``) against a payload re-computed at every launch
(``reflex.utils.js_runtimes._frontend_packages_cache_payload``). The payload
embeds machine-specific values (absolute bun path, package-manager probes) and
the operator's ports, so a marker frozen in a shipped payload NEVER matches the
end user's machine — and a miss means ``bun add`` runs and refuses to work
offline (manifest fetches; spike S4b, 2026-10-01). Worse, the fingerprint
includes pydantic's set-order-sensitive JSON of ``_non_default_attributes``,
so cross-process equality additionally requires a shared ``PYTHONHASHSEED``.
The caller generates ONE secret per-install seed (never a public constant — a
known seed on a network-reachable server is a hash-flooding invitation), stores
it in the per-user seed record, and passes it to BOTH this warmup and the real
run.

The warm trick: run the REAL reflex CLI entry (``python -m reflex run ...``,
same argv shape as the actual launch, so config loading + mutation order are
byte-identical) with exactly one patch — the install step is replaced by
'compute the fingerprint with reflex's own payload function, write the marker,
abort'. When the real run launches moments later, the marker matches and the
package manager is never invoked.

Private reflex internals are guarded: on any shape change (renamed/moved
helpers — we pin reflex via uv.lock, so this should not happen) we exit 3 and
the caller proceeds WITHOUT a marker, which degrades to today's behavior
(online install attempt), never worse.

Exit codes: 0 marker written (the install hook fired and was intercepted);
1 the reflex run aborted with a different SystemExit code (launch config
problem — real launch would fail too); 2 the install hook never fired (compile
failed or reflex stopped calling the install step); 3 reflex internals moved.
"""

from __future__ import annotations

import pickle  # nosec B403 — writes Reflex's OWN cache marker format locally; no untrusted input is unpickled
import runpy
import sys
from typing import Any

_HOOK_EXIT = 43


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: python -m backpropagate.ui_marker_warmup <port> <backend-host>",
            file=sys.stderr,
        )
        return 2
    port, backend_host = argv[1], argv[2]

    try:
        import reflex.utils.js_runtimes as js_runtimes
    except ImportError:
        return 3
    payload_fn = getattr(js_runtimes, "_frontend_packages_cache_payload", None)
    cache_path_fn = getattr(js_runtimes, "_frontend_packages_cache_path", None)
    if payload_fn is None or cache_path_fn is None:
        return 3

    def warm(*args: Any) -> Any:
        # Reflex's cached_procedure fingerprints the install call with the SAME arguments the
        # call receives, and that argument list moved between releases (0.9.5: packages, config,
        # managers; 0.9.12: packages, dev deps, frozen_lockfile, managers). Forwarding them
        # untouched keeps the marker byte-identical to the one reflex would write itself.
        marker = cache_path_fn()
        marker.parent.mkdir(parents=True, exist_ok=True)
        payload = payload_fn(*args)
        marker.write_bytes(pickle.dumps((payload, None)))
        print(f"ui_marker_warmup: wrote {marker}", flush=True)
        raise SystemExit(_HOOK_EXIT)

    js_runtimes._install_frontend_packages = warm  # type: ignore[assignment]  # the in-module call site reads the module global

    sys.argv = [
        "reflex",
        "run",
        "--env",
        "prod",
        "--frontend-port",
        port,
        "--backend-port",
        port,
        "--backend-host",
        backend_host,
    ]
    try:
        runpy.run_module("reflex", run_name="__main__")
    except SystemExit as exc:
        if exc.code == _HOOK_EXIT:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
