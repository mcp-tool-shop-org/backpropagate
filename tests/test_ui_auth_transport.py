"""Lane C transport checks: access-log query stripping and the Reflex CSP.

No server is started. The token in these tests is a fixed fake value so a
failure diff cannot contain a credential from the environment.
"""

from __future__ import annotations

import logging

from backpropagate.ui_app import access_log
from backpropagate.ui_security import _ui_listen_port, get_reflex_csp

_FAKE_QUERY_TOKEN = "uitoken-test-value"


def test_listen_port_falls_back_on_garbage(monkeypatch):
    monkeypatch.delenv("BACKPROPAGATE_UI_PORT", raising=False)
    assert _ui_listen_port({}) == 7862
    assert _ui_listen_port() == 7862
    assert _ui_listen_port({"BACKPROPAGATE_UI_PORT": "9000"}) == 9000
    assert _ui_listen_port({"BACKPROPAGATE_UI_PORT": "0"}) == 7862
    assert _ui_listen_port({"BACKPROPAGATE_UI_PORT": "70000"}) == 7862
    assert _ui_listen_port({"BACKPROPAGATE_UI_PORT": "nope"}) == 7862
    assert _ui_listen_port({"BACKPROPAGATE_UI_PORT": "  "}) == 7862


def test_connect_src_follows_the_listen_port_and_share_host(monkeypatch):
    monkeypatch.delenv("BACKPROPAGATE_UI_PORT", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_UI_SHARE_HOST", raising=False)
    policy = get_reflex_csp().build_policy()
    assert (
        "connect-src 'self' ws://127.0.0.1:7862 "
        "ws://localhost:7862 ws://[::1]:7862"
    ) in policy
    image_src = policy.split("img-src ", 1)[1].split(";", 1)[0]
    assert "https:" not in image_src

    monkeypatch.setenv("BACKPROPAGATE_UI_PORT", "9000")
    monkeypatch.setenv("BACKPROPAGATE_UI_SHARE_HOST", "Tunnel.Example")
    policy = get_reflex_csp().build_policy()
    assert "ws://127.0.0.1:9000" in policy
    assert "wss://tunnel.example" in policy
    assert "ws://tunnel.example" in policy


def test_access_filter_install_is_idempotent():
    access_log.install_access_query_filter()
    access_log.install_access_query_filter()
    logger = logging.getLogger("uvicorn.access")
    assert logger.filters.count(access_log._FILTER) == 1
    granian = logging.getLogger("granian.access")
    assert granian.filters.count(access_log._FILTER) == 1


def test_access_record_drops_the_query(caplog):
    access_log.install_access_query_filter()
    logger = logging.getLogger("uvicorn.access")
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="uvicorn.access"):
        logger.info(
            '%s - "%s" %s',
            "127.0.0.1",
            f"GET /runs?token={_FAKE_QUERY_TOKEN} HTTP/1.1",
            302,
        )
    assert _FAKE_QUERY_TOKEN not in caplog.text
    assert "/runs" in caplog.text


def test_access_filter_scrubs_args_and_keeps_format_strings():
    record = logging.LogRecord(
        name="granian.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg="%(path)s %(status)s",
        args={
            "path": f"/runs?token={_FAKE_QUERY_TOKEN}",
            "query_string": f"token={_FAKE_QUERY_TOKEN}".encode("ascii"),
            "query": f"token={_FAKE_QUERY_TOKEN}",
            "status": 302,
        },
        exc_info=None,
    )
    record.message = "cached"
    assert access_log._FILTER.filter(record) is True
    assert record.msg == "%(path)s %(status)s"
    assert record.args["query_string"] == b""
    assert record.args["query"] == ""
    assert _FAKE_QUERY_TOKEN not in record.args["path"]
    assert record.args["status"] == 302
    assert "message" not in record.__dict__
    assert _FAKE_QUERY_TOKEN not in record.getMessage()


def test_access_filter_scrubs_bytes_in_a_tuple():
    raw = f"GET /?token={_FAKE_QUERY_TOKEN}".encode("ascii")
    record = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg=f"GET /runs?token={_FAKE_QUERY_TOKEN}",
        args=(raw,),
        exc_info=None,
    )
    assert access_log._FILTER.filter(record) is True
    assert _FAKE_QUERY_TOKEN not in record.msg
    assert _FAKE_QUERY_TOKEN.encode("ascii") not in record.args[0]


def test_access_filter_ignores_a_record_without_args():
    record = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg="GET /runs", args=None, exc_info=None,
    )
    assert access_log._FILTER.filter(record) is True
    assert record.msg == "GET /runs"


def test_access_filter_failure_still_emits(caplog):
    class Boom:
        msg = "plain"

        @property
        def args(self):
            raise RuntimeError("nope")

    with caplog.at_level(logging.DEBUG, logger="backpropagate.ui_app.access_log"):
        assert access_log._FILTER.filter(Boom()) is True
    assert "RuntimeError" in caplog.text
