"""Strip query strings from UI access logs.

``backprop ui`` starts ``python -m reflex run --env prod``. On Windows that
server is Granian (access log off unless enabled) or uvicorn (the access
line includes the query). Either way the launch token must not land in an
access record. The server process imports this app after its own
``dictConfig``, which replaces handlers but leaves logger filters in place,
so installing the filter at import — and again on each request — is the
hook that actually runs.
"""

from __future__ import annotations

import logging

_ACCESS_LOGGERS = ("uvicorn.access", "granian.access")
_QUERY_FIELDS = frozenset({"query_string", "qs", "query"})


def _strip_text(value: str) -> str:
    """Drop a query string so a ``?token=`` tail cannot be formatted later."""
    cut = value.find("?")
    if cut == -1:
        return value
    return value[:cut]


def _scrub_value(value: object) -> object:
    if isinstance(value, str):
        return _strip_text(value)
    if isinstance(value, bytes):
        text = value.decode("latin-1", "replace")
        return _strip_text(text).encode("latin-1", "replace")
    return value


def _looks_like_format(msg: str) -> bool:
    return "%s" in msg or "%d" in msg or "%(" in msg


class _AccessQueryFilter(logging.Filter):
    """Remove query strings from uvicorn and granian access records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            self._scrub(record)
        except Exception as exc:  # noqa: BLE001 — a filter must not break the server
            logging.getLogger(__name__).debug(
                "access log query strip failed: %s", type(exc).__name__,
            )
        return True

    def _scrub(self, record: logging.LogRecord) -> None:
        args = record.args
        if isinstance(args, dict):
            cleaned: dict[str, object] = {}
            for key, value in args.items():
                if key in _QUERY_FIELDS:
                    cleaned[key] = b"" if isinstance(value, bytes) else ""
                else:
                    cleaned[key] = _scrub_value(value)
            record.args = cleaned
        elif isinstance(args, tuple):
            record.args = tuple(_scrub_value(item) for item in args)
        msg = record.msg
        if isinstance(msg, str) and "?" in msg and not _looks_like_format(msg):
            record.msg = _strip_text(msg)
        # getMessage caches on the record. Drop a stale cache so the handler
        # formats the scrubbed args, not the line it already built.
        if "message" in record.__dict__:
            del record.message


_FILTER = _AccessQueryFilter()


def install_access_query_filter() -> None:
    """Attach the query-stripping filter to the UI server access loggers.

    Idempotent. Safe to call at import and again on each request: a later
    ``dictConfig`` replaces handlers but does not clear filters, and a
    repeat call does not stack filters.
    """
    for name in _ACCESS_LOGGERS:
        logger = logging.getLogger(name)
        if _FILTER not in logger.filters:
            logger.addFilter(_FILTER)
