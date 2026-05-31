"""
Backpropagate - Dataset Quality Report (v1.5 T1.1)
==================================================

A cheap, **torch-free** data-quality report for fine-tuning datasets. This is
the "next moat" feature from ``docs/V1_5_BRIEF.md`` §4 (T1.1): operators'
number-one unmet pain is dataset prep + eval, not the training loop. The
report composes the existing :mod:`backpropagate.datasets` primitives
(``detect_format`` / ``validate_dataset`` / ``deduplicate_exact`` /
``_count_tokens_approx`` / ``_get_ngrams`` / ``convert_to_chatml``) into a
single pass that surfaces:

- format distribution (per-sample auto-detect)
- exact duplicates + near-duplicate clusters (deterministic MinHash + LSH)
- token-length histogram (over ChatML-converted rows)
- length outliers (mean ± sigma, plus a hard absolute cap)
- empty / near-empty turns (the ``_CHATML_TURN_RE`` contract)
- rows with no assistant response
- optional train/test contamination flags (overlap vs a held-out set)

and a PASS / WARN / FAIL verdict with named tripped gates.

Import discipline
-----------------
This module mirrors the import discipline of ``cli.cmd_validate``: it does
**not** import ``torch`` and pulls only the lightweight detect/dedupe/convert
helpers from :mod:`backpropagate.datasets`. (Importing the package at all pays
the package-level torch cost via ``backpropagate/__init__``; this module adds
none of its own and does no model loading — analysis is pure-Python.)

Determinism
-----------
Same input list -> byte-identical report. The near-duplicate detector is a
hand-rolled MinHash + banding LSH built on ``hashlib.blake2b`` (no
``datasketch`` dependency, no ``random`` module): the permutation seeds are a
fixed, deterministically generated table, so re-running on the same samples
yields identical cluster counts and rates. This matches the determinism
contract already documented on ``datasets.deduplicate_minhash``.

Usage::

    from backpropagate.dataset_report import analyze_dataset

    report = analyze_dataset(samples, fail_on_dups=0.1)
    print(report.summary())
    if report.verdict == "FAIL":
        ...  # CI gate
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from .datasets import (
    _CHATML_TURN_RE,
    DatasetFormat,
    _count_tokens_approx,
    _extract_think_spans,
    _get_ngrams,
    convert_to_chatml,
    detect_format,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ContaminationResult",
    "DataQualityReport",
    "analyze_dataset",
    "find_duplicate_clusters",
    "token_length_histogram",
    "trace_length_histogram",
    "contamination_overlap",
]


# =============================================================================
# TUNABLES
# =============================================================================

# Length outliers: a row is an outlier if it sits beyond mean ± sigma*stdev OR
# above this absolute token cap. The absolute cap catches a single pathological
# 50k-token row even in a corpus whose stdev is large enough that 3 sigma would
# not flag it (e.g. a tiny sample where one giant row dominates the variance).
_ABSOLUTE_TOKEN_CAP = 16_384

# Strict-preset advisory→hard thresholds (T1.1 contract).
_STRICT_DUP_RATE = 0.25
_STRICT_CONTAM_RATE = 0.01
_STRICT_EMPTY_RATE = 0.05

# Advisory (WARN) thresholds — tripping these never FAILs, only annotates.
_WARN_DUP_RATE = 0.10
_WARN_EMPTY_RATE = 0.02
_WARN_NO_ASSISTANT_RATE = 0.02
_WARN_OUTLIER_RATE = 0.05

# MinHash / LSH configuration. Banding (b bands × r rows) approximates a
# Jaccard threshold of ~ (1/b) ** (1/r). With num_perm derived from b*r we hit
# the requested threshold closely while staying deterministic.
_DEFAULT_NUM_PERM = 128
_MINHASH_PRIME = (1 << 61) - 1  # Mersenne prime, standard MinHash modulus
_MINHASH_MAX_HASH = (1 << 32) - 1


# =============================================================================
# DATACLASSES (FIXED PUBLIC CONTRACT — do not change field names/order)
# =============================================================================


@dataclass
class ContaminationResult:
    """Train/test overlap result from :func:`contamination_overlap`."""

    overlap_rows: int
    overlap_rate: float
    against_path: str


@dataclass
class DataQualityReport:
    """The full data-quality report returned by :func:`analyze_dataset`."""

    total_rows: int
    parseable_rows: int
    parse_errors: int
    format_distribution: dict[str, int]
    exact_duplicates: int
    duplicate_clusters: int
    near_duplicate_rate: float
    empty_turn_rows: int
    outlier_rows: int
    no_assistant_rows: int
    token_histogram: list[tuple[int, int]]
    contamination: ContaminationResult | None
    verdict: str
    failed_thresholds: list[str]
    # Reasoning-trace fields (v1.5 T3.2) — APPENDED after the existing contract
    # (never reordered). ``think_rows`` is the count of rows carrying at least
    # one ``<think>`` span; ``think_pct`` is that fraction of total rows;
    # ``trace_histogram`` buckets those rows by summed think-span token length
    # (see :func:`trace_length_histogram`). On a non-reasoning dataset these are
    # ``0`` / ``0.0`` / an all-zero histogram and the summary stays silent.
    think_rows: int = 0
    think_pct: float = 0.0
    trace_histogram: list[tuple[int, int]] = field(default_factory=list)

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        lines = [
            "Dataset Quality Report",
            "=" * 40,
            f"Verdict: {self.verdict}",
            f"Total rows: {self.total_rows}",
            f"Parseable rows: {self.parseable_rows}",
        ]
        if self.parse_errors:
            lines.append(f"Parse errors: {self.parse_errors}")

        if self.format_distribution:
            dist = ", ".join(
                f"{fmt}={count}"
                for fmt, count in sorted(self.format_distribution.items())
            )
            lines.append(f"Formats: {dist}")

        lines.extend(
            [
                "",
                "Duplicates:",
                f"  Exact duplicates: {self.exact_duplicates}",
                f"  Near-duplicate clusters: {self.duplicate_clusters}",
                f"  Near-duplicate rate: {100 * self.near_duplicate_rate:.1f}%",
                "",
                "Quality flags:",
                f"  Empty-turn rows: {self.empty_turn_rows}",
                f"  Length outliers: {self.outlier_rows}",
                f"  No-assistant rows: {self.no_assistant_rows}",
            ]
        )

        if self.token_histogram:
            lines.append("")
            lines.append("Token-length histogram (ChatML-converted):")
            for upper, count in self.token_histogram:
                label = "inf" if upper < 0 else f"<={upper}"
                lines.append(f"  {label:>8}: {count}")
            # Surface the documented CJK under-count caveat from
            # datasets._count_tokens_approx so operators don't mis-read the
            # histogram on a non-ASCII corpus (CJK is ~1+ token/char, so the
            # ~4 chars/token estimate UNDER-counts it — the histogram skews
            # SHORT on CJK).
            lines.append(
                "  (note: token counts are a ~4 chars/token approximation and "
                "under-count CJK by ~4-8x; re-derive against your real "
                "tokenizer for non-ASCII-English data.)"
            )

        # Reasoning-trace block (v1.5 T3.2): only shown when the dataset
        # actually carries <think> traces, so non-reasoning datasets (e.g. a
        # plain contamination check) stay quiet.
        if self.think_rows > 0:
            lines.extend(
                [
                    "",
                    "Reasoning traces:",
                    f"  Rows with <think>: {self.think_rows} "
                    f"({100 * self.think_pct:.1f}%)",
                ]
            )
            if self.trace_histogram:
                lines.append("  Trace-length histogram (think-span tokens):")
                for upper, count in self.trace_histogram:
                    label = "inf" if upper < 0 else f"<={upper}"
                    lines.append(f"    {label:>8}: {count}")

        if self.contamination is not None:
            lines.extend(
                [
                    "",
                    "Contamination:",
                    f"  vs {self.contamination.against_path}: "
                    f"{self.contamination.overlap_rows} rows "
                    f"({100 * self.contamination.overlap_rate:.1f}%)",
                ]
            )

        if self.failed_thresholds:
            lines.append("")
            lines.append("Tripped gates:")
            for gate in self.failed_thresholds:
                lines.append(f"  - {gate}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (tuples -> lists, nested dataclass -> dict)."""
        data = asdict(self)
        # asdict already turned the nested ContaminationResult dataclass into a
        # dict and left None as None. Normalise the histogram tuples to lists so
        # the payload round-trips through json.dumps/loads unchanged.
        data["token_histogram"] = [
            [upper, count] for upper, count in self.token_histogram
        ]
        # Same tuple->list normalisation for the appended reasoning-trace
        # histogram so the whole payload round-trips through json.dumps/loads.
        data["trace_histogram"] = [
            [upper, count] for upper, count in self.trace_histogram
        ]
        return data


# =============================================================================
# DETERMINISTIC MINHASH + LSH (no datasketch, no random)
# =============================================================================


def _stable_hash32(data: bytes) -> int:
    """Deterministic 32-bit hash of ``data`` via blake2b.

    Python's builtin ``hash()`` is salted per-process (PYTHONHASHSEED) and so is
    NOT reproducible across runs; the dedup determinism contract requires a
    stable hash. blake2b is in the stdlib and deterministic.
    """
    digest = hashlib.blake2b(data, digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _permutation_table(num_perm: int) -> list[tuple[int, int]]:
    """Build a fixed (a, b) coefficient table for the MinHash permutations.

    The coefficients are derived deterministically from a constant seed string
    via blake2b, so the table is identical on every run / process / platform —
    the source of the determinism guarantee. ``a`` is forced odd and non-zero
    (a requirement for the multiplier in a universal hash mod a prime).
    """
    table: list[tuple[int, int]] = []
    for i in range(num_perm):
        a_bytes = hashlib.blake2b(
            f"backpropagate-minhash-a-{i}".encode(), digest_size=8
        ).digest()
        b_bytes = hashlib.blake2b(
            f"backpropagate-minhash-b-{i}".encode(), digest_size=8
        ).digest()
        a = int.from_bytes(a_bytes, "big") % _MINHASH_PRIME
        b = int.from_bytes(b_bytes, "big") % _MINHASH_PRIME
        if a == 0:
            a = 1
        a |= 1  # force odd
        table.append((a, b))
    return table


def _minhash_signature(
    grams: list[str], perms: list[tuple[int, int]]
) -> tuple[int, ...] | None:
    """Compute the MinHash signature for a set of n-grams.

    Returns ``None`` when there are no grams (empty/whitespace content) — such
    rows have no comparable signature and must never be folded together (the
    same special-case ``datasets._get_ngrams`` / ``deduplicate_minhash`` make).
    """
    if not grams:
        return None
    # Deduplicate grams to a set of base hashes first (Jaccard is over the SET
    # of shingles, and it makes the inner loop cheaper).
    base_hashes = {_stable_hash32(g.encode("utf-8")) for g in grams}
    signature: list[int] = []
    for a, b in perms:
        best = _MINHASH_MAX_HASH
        for h in base_hashes:
            val = ((a * h + b) % _MINHASH_PRIME) & _MINHASH_MAX_HASH
            if val < best:
                best = val
        signature.append(best)
    return tuple(signature)


def _choose_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows_per_band) whose LSH S-curve threshold ≈ ``threshold``.

    The probability two items share a band is ~ 1 - (1 - s**r) ** b, whose
    inflection ("threshold") is approximately (1/b) ** (1/r). We scan every
    factorisation b*r == num_perm and keep the one whose approximate threshold
    is closest to the requested Jaccard threshold. Deterministic.
    """
    best: tuple[int, int] | None = None
    best_err = float("inf")
    for r in range(1, num_perm + 1):
        if num_perm % r != 0:
            continue
        b = num_perm // r
        approx = (1.0 / b) ** (1.0 / r)
        err = abs(approx - threshold)
        if err < best_err:
            best_err = err
            best = (b, r)
    # num_perm >= 1 guarantees at least the (b=num_perm, r=1) factorisation.
    assert best is not None
    return best


def _union_find_clusters(
    signatures: list[tuple[int, ...] | None],
    *,
    bands: int,
    rows_per_band: int,
) -> list[int]:
    """Group rows into near-duplicate clusters via banding LSH + union-find.

    Returns a parent array (DSU roots). Rows with a ``None`` signature
    (empty content) are left as singletons — never unioned with anything.
    """
    n = len(signatures)
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression for speed; deterministic outcome either way.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # Deterministic merge: always attach the larger root index under the
        # smaller, so the representative is order-independent.
        lo, hi = (rx, ry) if rx < ry else (ry, rx)
        parent[hi] = lo

    # For each band, bucket rows by their slice of the signature; any two rows
    # sharing a bucket in any band are candidate near-duplicates and get unioned.
    for band in range(bands):
        start = band * rows_per_band
        end = start + rows_per_band
        buckets: dict[tuple[int, ...], int] = {}
        for idx, sig in enumerate(signatures):
            if sig is None:
                continue
            key = sig[start:end]
            first = buckets.get(key)
            if first is None:
                buckets[key] = idx
            else:
                union(first, idx)

    return [find(i) for i in range(n)]


# =============================================================================
# PUBLIC PRIMITIVES
# =============================================================================


def find_duplicate_clusters(
    samples: list[dict],
    *,
    key: str = "text",
    threshold: float = 0.9,
    num_perm: int = _DEFAULT_NUM_PERM,
) -> tuple[int, int, float]:
    """Find near-duplicate clusters in ``samples``.

    Reuses ``datasets._get_ngrams`` for shingling and a deterministic MinHash +
    banding LSH (no datasketch) for membership, so the result is reproducible
    across runs (the determinism contract).

    Args:
        samples: List of samples (dicts with a text ``key``, or strings).
        key: Field holding the text when samples are dicts.
        threshold: Jaccard similarity threshold for "near-duplicate" (0..1).
        num_perm: Number of MinHash permutations.

    Returns:
        ``(num_clusters, exact_duplicates, near_duplicate_rate)`` where:
        - ``num_clusters`` is the number of multi-member near-duplicate clusters
          (clusters of size >= 2; singletons are not counted).
        - ``exact_duplicates`` is the count of rows that are byte-identical to an
          earlier row (the classic "rows removed by exact dedup").
        - ``near_duplicate_rate`` is the fraction of rows that are a redundant
          member of some near-duplicate cluster, i.e.
          ``(total_rows - num_unique_clusters_and_singletons) / total_rows``.
    """
    n = len(samples)
    if n == 0:
        return 0, 0, 0.0

    texts: list[str] = [_resolve_text(s, key) for s in samples]

    # Exact duplicates: count rows identical to an earlier row (stable hash so
    # the count is deterministic and matches deduplicate_exact's "num_removed").
    seen: set[int] = set()
    exact_duplicates = 0
    for t in texts:
        h = _stable_hash32(t.encode("utf-8"))
        if h in seen:
            exact_duplicates += 1
        else:
            seen.add(h)

    perms = _permutation_table(num_perm)
    signatures = [
        _minhash_signature(_get_ngrams(t, n=3), perms) for t in texts
    ]

    bands, rows_per_band = _choose_bands(num_perm, threshold)
    roots = _union_find_clusters(
        signatures, bands=bands, rows_per_band=rows_per_band
    )

    # Tally cluster sizes by root.
    sizes: dict[int, int] = {}
    for r in roots:
        sizes[r] = sizes.get(r, 0) + 1

    num_clusters = sum(1 for size in sizes.values() if size >= 2)
    # Redundant rows = total minus the number of distinct groups (each group
    # keeps exactly one canonical representative). Singletons contribute 0.
    redundant = sum(size - 1 for size in sizes.values())
    near_duplicate_rate = redundant / n if n else 0.0

    return num_clusters, exact_duplicates, near_duplicate_rate


def token_length_histogram(
    samples: list[dict],
    *,
    bins: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096, 8192),
) -> list[tuple[int, int]]:
    """Bucket samples by approximate ChatML token length.

    Each sample is converted to ChatML (per-sample format auto-detect, reusing
    ``datasets.convert_to_chatml``) and counted with
    ``datasets._count_tokens_approx``. Rows that fail to convert are skipped
    (they are counted as parse errors elsewhere).

    Args:
        samples: List of samples in any supported format.
        bins: Ascending bucket upper bounds. A trailing "infinity" bucket is
            always appended with an upper bound of ``-1``.

    Returns:
        ``[(upper_bound, count), ...]`` with one entry per bin plus a final
        ``(-1, count)`` overflow bucket for rows longer than the last bin.
    """
    sorted_bins = tuple(sorted(bins))
    counts = [0] * (len(sorted_bins) + 1)  # +1 for the overflow ("inf") bucket

    for s in samples:
        chatml = _to_chatml_text(s)
        if chatml is None:
            continue
        tokens = _count_tokens_approx(chatml)
        placed = False
        for i, upper in enumerate(sorted_bins):
            if tokens <= upper:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    histogram: list[tuple[int, int]] = [
        (upper, counts[i]) for i, upper in enumerate(sorted_bins)
    ]
    histogram.append((-1, counts[-1]))  # -1 upper bound == "inf"
    return histogram


def trace_length_histogram(
    samples: list[dict],
    *,
    bins: tuple[int, ...] = (8, 32, 128, 512, 2048, 8192),
) -> list[tuple[int, int]]:
    """Bucket samples by the token length of their ``<think>`` reasoning trace.

    Reasoning-trace SFT companion to :func:`token_length_histogram` (v1.5 T3.2).
    Each sample is converted to ChatML (per-sample auto-detect, reusing
    ``datasets.convert_to_chatml``); the token counts of ALL its ``<think>``
    spans (``datasets._extract_think_spans``) are summed with
    ``datasets._count_tokens_approx``. **Rows with NO ``<think>`` span are
    skipped entirely** — they are not reasoning rows and a 0-length bucket would
    swamp the histogram on a mixed corpus. Unconvertible rows are also skipped.

    Args:
        samples: List of samples in any supported format.
        bins: Ascending bucket upper bounds (token counts). A trailing
            "infinity" bucket is always appended with an upper bound of ``-1``.

    Returns:
        ``[(upper_bound, count), ...]`` with one entry per bin plus a final
        ``(-1, count)`` overflow bucket for traces longer than the last bin. The
        counts sum to the number of rows that carry at least one ``<think>``
        span (NOT to the total row count). Token counts share the
        ``_count_tokens_approx`` ~4 chars/token caveat (under-counts CJK by
        ~4-8x, so the trace histogram skews short on CJK reasoning data).
    """
    sorted_bins = tuple(sorted(bins))
    counts = [0] * (len(sorted_bins) + 1)  # +1 for the overflow ("inf") bucket

    for s in samples:
        chatml = _to_chatml_text(s)
        if chatml is None:
            continue
        spans = _extract_think_spans(chatml)
        if not spans:
            # Not a reasoning row — excluded from the trace histogram.
            continue
        tokens = sum(_count_tokens_approx(span) for span in spans)
        placed = False
        for i, upper in enumerate(sorted_bins):
            if tokens <= upper:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    histogram: list[tuple[int, int]] = [
        (upper, counts[i]) for i, upper in enumerate(sorted_bins)
    ]
    histogram.append((-1, counts[-1]))  # -1 upper bound == "inf"
    return histogram


def contamination_overlap(
    train_samples: list[dict],
    test_samples: list[dict],
    *,
    threshold: float = 0.9,
) -> ContaminationResult:
    """Estimate how many train rows overlap a held-out/test set.

    A train row is "contaminated" if it is an exact duplicate of a test row OR a
    near-duplicate (Jaccard >= ``threshold`` via the same deterministic MinHash
    used elsewhere). The rate is over the TRAIN set (``overlap_rows / len(train)``).

    Args:
        train_samples: The training rows.
        test_samples: The held-out / evaluation rows to check against.
        threshold: Jaccard threshold for near-duplicate contamination.

    Returns:
        A :class:`ContaminationResult` (``against_path`` is left as the empty
        string here; :func:`analyze_dataset` fills in the human label).
    """
    n_train = len(train_samples)
    if n_train == 0 or not test_samples:
        return ContaminationResult(
            overlap_rows=0, overlap_rate=0.0, against_path=""
        )

    train_texts = [_resolve_text(s) for s in train_samples]
    test_texts = [_resolve_text(s) for s in test_samples]

    # Exact-overlap fast path via stable hashes.
    test_hashes = {_stable_hash32(t.encode("utf-8")) for t in test_texts}

    perms = _permutation_table(_DEFAULT_NUM_PERM)
    test_sigs = [
        _minhash_signature(_get_ngrams(t, n=3), perms) for t in test_texts
    ]
    bands, rows_per_band = _choose_bands(_DEFAULT_NUM_PERM, threshold)

    # Build banding buckets over the TEST signatures so each train row can be
    # checked in O(bands) instead of O(len(test)).
    band_buckets: list[dict[tuple[int, ...], list[int]]] = [
        {} for _ in range(bands)
    ]
    for t_idx, sig in enumerate(test_sigs):
        if sig is None:
            continue
        for band in range(bands):
            start = band * rows_per_band
            key = sig[start : start + rows_per_band]
            band_buckets[band].setdefault(key, []).append(t_idx)

    overlap_rows = 0
    for t in train_texts:
        h = _stable_hash32(t.encode("utf-8"))
        if h in test_hashes:
            overlap_rows += 1
            continue
        sig = _minhash_signature(_get_ngrams(t, n=3), perms)
        if sig is None:
            continue
        # A shared band bucket is a near-duplicate candidate. The banding
        # S-curve is calibrated to ``threshold`` so a candidate is treated as a
        # hit (consistent with find_duplicate_clusters, which unions on a shared
        # bucket without a second verification pass).
        hit = False
        for band in range(bands):
            start = band * rows_per_band
            key = sig[start : start + rows_per_band]
            if key in band_buckets[band]:
                hit = True
                break
        if hit:
            overlap_rows += 1

    return ContaminationResult(
        overlap_rows=overlap_rows,
        overlap_rate=overlap_rows / n_train,
        against_path="",
    )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _resolve_text(sample: dict | str, key: str = "text") -> str:
    """Resolve a sample to comparable text for dedup/contamination.

    Fast path: a string, or a dict carrying a non-empty ``key`` (already-ChatML
    rows), is used verbatim. Otherwise the sample is ChatML-converted so a raw
    Alpaca/ShareGPT/OpenAI dict (which has no ``text`` key) still compares on the
    same normalised text the trainer would see — without this fallback every
    such row resolved to ``""`` and collapsed into one giant false cluster.
    """
    if isinstance(sample, str):
        return sample
    val = sample.get(key)
    if isinstance(val, str) and val:
        return val
    return _to_chatml_text(sample) or ""


def _to_chatml_text(sample: dict | str) -> str | None:
    """Convert a single sample to ChatML text, or ``None`` if it can't convert.

    Reuses ``datasets.convert_to_chatml`` (which does per-sample format
    detection) so the histogram / empty-turn / no-assistant signals all see the
    SAME converted text the trainer would.
    """
    converted = convert_to_chatml([sample])
    if not converted:
        return None
    return converted[0].get("text", "")


def _has_empty_turn(chatml: str) -> bool:
    """True if any user/assistant turn body is blank (the _CHATML_TURN_RE contract)."""
    for role, body in _CHATML_TURN_RE.findall(chatml):
        if role in ("user", "assistant") and not body.strip():
            return True
    return False


def _has_assistant(chatml: str) -> bool:
    """True if the converted ChatML contains a non-empty assistant turn."""
    for role, body in _CHATML_TURN_RE.findall(chatml):
        if role == "assistant" and body.strip():
            return True
    return False


# =============================================================================
# THE REPORT
# =============================================================================


def analyze_dataset(
    samples: list[dict],
    *,
    format_hint: str | None = None,
    dup_threshold: float = 0.9,
    outlier_sigma: float = 3.0,
    against: list[dict] | None = None,
    against_path: str | None = None,
    fail_on_dups: float | None = None,
    fail_on_contamination: float | None = None,
    max_outlier_rate: float | None = None,
    strict: bool = False,
) -> DataQualityReport:
    """Compose the dataset-quality primitives into a single report.

    See module docstring for the signal list. ``format_hint`` (if given) names a
    :class:`~backpropagate.datasets.DatasetFormat` value (e.g. ``"alpaca"``) and
    is recorded in the format distribution for rows that detect as UNKNOWN; it
    does not override successful per-sample detection.

    The ``strict`` preset promotes three advisory signals to hard gates:
    near-duplicate rate > 0.25, contamination > 0.01, empty-turn rate > 0.05.
    Explicit ``fail_on_*`` / ``max_outlier_rate`` arguments are always honoured
    and stack with ``strict``.
    """
    total_rows = len(samples)

    # --- format distribution + parseability -------------------------------
    format_counts: dict[str, int] = {}
    parse_errors = 0
    hint_fmt = _coerce_format_hint(format_hint)

    chatml_texts: list[str | None] = []
    for s in samples:
        fmt = detect_format(s)
        if fmt == DatasetFormat.UNKNOWN and hint_fmt is not None:
            fmt = hint_fmt
        chatml = _to_chatml_text(s)
        if chatml is None or fmt == DatasetFormat.UNKNOWN:
            # Could not parse/convert this row into a usable training example.
            parse_errors += 1
            chatml_texts.append(None)
            # Still record the (unknown) format so the distribution sums to
            # total_rows.
            format_counts[fmt.value] = format_counts.get(fmt.value, 0) + 1
            continue
        format_counts[fmt.value] = format_counts.get(fmt.value, 0) + 1
        chatml_texts.append(chatml)

    parseable_rows = total_rows - parse_errors

    # --- duplicate analysis (over CONVERTED ChatML text) -------------------
    # Dedup/contamination must compare the SAME normalised text the trainer
    # sees, not a raw "text" key the source rows (Alpaca/ShareGPT/OpenAI) don't
    # have. Unconvertible rows become empty text, which _get_ngrams treats as
    # no-content (kept as singletons, never folded together).
    converted_rows: list[dict[str, str]] = [
        {"text": c if c is not None else ""} for c in chatml_texts
    ]
    duplicate_clusters, exact_duplicates, near_duplicate_rate = (
        find_duplicate_clusters(
            converted_rows, key="text", threshold=dup_threshold
        )
    )

    # --- token histogram (over ALL samples; skips unconvertible internally) -
    token_histogram = token_length_histogram(samples)

    # --- per-row quality flags over the converted ChatML -------------------
    empty_turn_rows = 0
    no_assistant_rows = 0
    think_rows = 0
    token_lengths: list[int] = []
    for chatml in chatml_texts:
        if chatml is None:
            continue
        if _has_empty_turn(chatml):
            empty_turn_rows += 1
        if not _has_assistant(chatml):
            no_assistant_rows += 1
        # Reasoning-trace tally (v1.5 T3.2): a row "has a trace" when it carries
        # at least one <think>...</think> span in its converted ChatML.
        if _extract_think_spans(chatml):
            think_rows += 1
        token_lengths.append(_count_tokens_approx(chatml))

    # think_pct is over TOTAL rows (matching the histogram's row universe and
    # the summary's "(X%)" reading); 0.0 on an empty dataset.
    think_pct = think_rows / total_rows if total_rows else 0.0
    trace_histogram = trace_length_histogram(samples)

    # --- length outliers ---------------------------------------------------
    outlier_rows = _count_length_outliers(token_lengths, sigma=outlier_sigma)

    # --- contamination -----------------------------------------------------
    # Convert the held-out rows the same way before comparing, so train/test
    # overlap is measured on normalised ChatML (matching the dedup pass above).
    contamination: ContaminationResult | None = None
    if against is not None:
        against_rows: list[dict[str, str]] = [
            {"text": _to_chatml_text(s) or ""} for s in against
        ]
        contamination = contamination_overlap(
            converted_rows, against_rows, threshold=dup_threshold
        )
        contamination.against_path = against_path or "held-out set"

    # --- verdict + gates ---------------------------------------------------
    failed_thresholds: list[str] = []
    warn_flags = False

    # Effective hard thresholds (strict preset OR explicit args; explicit wins
    # by being the min when both apply, i.e. the stricter of the two).
    eff_fail_dups = _tighter(fail_on_dups, _STRICT_DUP_RATE if strict else None)
    eff_fail_contam = _tighter(
        fail_on_contamination, _STRICT_CONTAM_RATE if strict else None
    )
    eff_empty_gate = _STRICT_EMPTY_RATE if strict else None

    empty_rate = empty_turn_rows / parseable_rows if parseable_rows else 0.0
    no_assistant_rate = (
        no_assistant_rows / parseable_rows if parseable_rows else 0.0
    )
    outlier_rate = outlier_rows / parseable_rows if parseable_rows else 0.0

    # Hard gates -> FAIL.
    if eff_fail_dups is not None and near_duplicate_rate > eff_fail_dups:
        failed_thresholds.append(
            f"near_duplicate_rate {near_duplicate_rate:.3f} > "
            f"fail_on_dups {eff_fail_dups:.3f}"
        )
    if (
        eff_fail_contam is not None
        and contamination is not None
        and contamination.overlap_rate > eff_fail_contam
    ):
        failed_thresholds.append(
            f"contamination_rate {contamination.overlap_rate:.3f} > "
            f"fail_on_contamination {eff_fail_contam:.3f}"
        )
    if max_outlier_rate is not None and outlier_rate > max_outlier_rate:
        failed_thresholds.append(
            f"outlier_rate {outlier_rate:.3f} > "
            f"max_outlier_rate {max_outlier_rate:.3f}"
        )
    if eff_empty_gate is not None and empty_rate > eff_empty_gate:
        failed_thresholds.append(
            f"empty_turn_rate {empty_rate:.3f} > "
            f"strict empty gate {eff_empty_gate:.3f}"
        )

    # Advisory (WARN) signals — present but not gated.
    if parse_errors > 0:
        warn_flags = True
    if near_duplicate_rate > _WARN_DUP_RATE:
        warn_flags = True
    if empty_rate > _WARN_EMPTY_RATE:
        warn_flags = True
    if no_assistant_rate > _WARN_NO_ASSISTANT_RATE:
        warn_flags = True
    if outlier_rate > _WARN_OUTLIER_RATE:
        warn_flags = True
    if contamination is not None and contamination.overlap_rows > 0:
        warn_flags = True

    if failed_thresholds:
        verdict = "FAIL"
    elif warn_flags:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return DataQualityReport(
        total_rows=total_rows,
        parseable_rows=parseable_rows,
        parse_errors=parse_errors,
        format_distribution=format_counts,
        exact_duplicates=exact_duplicates,
        duplicate_clusters=duplicate_clusters,
        near_duplicate_rate=near_duplicate_rate,
        empty_turn_rows=empty_turn_rows,
        outlier_rows=outlier_rows,
        no_assistant_rows=no_assistant_rows,
        token_histogram=token_histogram,
        contamination=contamination,
        verdict=verdict,
        failed_thresholds=failed_thresholds,
        think_rows=think_rows,
        think_pct=think_pct,
        trace_histogram=trace_histogram,
    )


def _coerce_format_hint(format_hint: str | None) -> DatasetFormat | None:
    """Map a string hint to a DatasetFormat, tolerating unknown labels."""
    if format_hint is None:
        return None
    try:
        return DatasetFormat(format_hint.lower())
    except ValueError:
        logger.warning(
            "analyze_dataset: unrecognized format_hint %r (ignored); "
            "valid values: %s",
            format_hint,
            ", ".join(f.value for f in DatasetFormat),
        )
        return None


def _tighter(a: float | None, b: float | None) -> float | None:
    """Return the stricter (smaller) of two optional thresholds."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _count_length_outliers(token_lengths: list[int], *, sigma: float) -> int:
    """Count rows beyond mean ± sigma*stdev OR over the absolute token cap.

    The absolute cap is a backstop: in a small corpus a single 50k-token row
    inflates the stdev enough that ``mean + 3*stdev`` can exceed the row itself,
    so the statistical test alone would miss it. The cap guarantees a single
    pathological row is always flagged.
    """
    n = len(token_lengths)
    if n == 0:
        return 0
    if n < 2:
        # Can't compute a meaningful stdev; only the absolute cap applies.
        return sum(1 for t in token_lengths if t > _ABSOLUTE_TOKEN_CAP)

    mean = statistics.fmean(token_lengths)
    stdev = statistics.pstdev(token_lengths)
    high = mean + sigma * stdev
    low = mean - sigma * stdev

    outliers = 0
    for t in token_lengths:
        if t > _ABSOLUTE_TOKEN_CAP or t > high or t < low:
            outliers += 1
    return outliers
