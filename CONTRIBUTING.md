# Contributing to Backpropagate

Thank you for your interest in contributing to Backpropagate. This document covers the local dev loop, what a "good first contribution" looks like, the swarm-and-amend cadence we use for larger work, and the issue / PR conventions.

## What a good first contribution looks like

- **Fix a Stage-A or Stage-B finding from a dogfood-swarm audit.** Each release has a `swarms/` JSON corpus under `dogfood-labs/swarms/`; pick a MEDIUM/LOW item nobody's claimed and submit a PR with the fix + a targeted regression test. Wave audit JSON is the canonical "good first issue" board for this repo.
- **Add a recipe to the handbook.** If you used Backpropagate to do something the [Recipes page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/) doesn't yet cover, write up the paste-and-run version. Even a 15-line recipe + 4 sentences of context is high-leverage.
- **Add a test for a corner you noticed wasn't covered.** Coverage floor is 50%, but several files sit at 60–80% and there are real gaps. `pytest --cov=backpropagate --cov-report=html` then `open htmlcov/index.html` shows you which lines are uncovered.
- **Triage a stale issue or open Discussion.** Repro the bug, attach a minimal example, and confirm the error code — that drops the maintainer's triage cost dramatically.

If you're not sure what to pick up, start a thread in [Discussions → Ideas](https://github.com/mcp-tool-shop-org/backpropagate/discussions/categories/ideas) and the maintainer will point at something matched to your interests.

## Development Setup

### Prerequisites

- Python 3.10+ (3.11 is the most-tested floor; 3.10 reaches upstream EOL October 2026 and will be dropped in v1.4)
- CUDA-capable GPU (for full testing — many tests are CPU-only and run fine on macOS / non-GPU Linux for triage)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/mcp-tool-shop-org/backpropagate
cd backpropagate

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode with all dependencies
pip install -e ".[dev,full]"

# Install pre-commit hooks
pre-commit install
```

### The local dev loop

The four commands the CI runs on every PR — run them locally before pushing and you'll catch ~95% of CI failures:

```bash
ruff check backpropagate/                                   # lint (~1s)
mypy backpropagate/ --ignore-missing-imports                # type check (~10s)
pytest tests/ -m "not gpu and not slow and not integration" # tests (~30s on a 16-core box)
python scripts/check_doc_drift.py                           # doc-drift gate (~1s)
```

The full suite (`pytest tests/`) runs ~2000 tests in 30–60 seconds; the `not gpu and not slow and not integration` filter skips ~20 GPU-bound tests that need a CUDA card. For coverage: `pytest --cov=backpropagate --cov-report=term-missing tests/` (or `html` for the rendered surface). If you just touched one file, `pytest tests/test_<that_file>.py` runs in <5 seconds.

The drift gate is load-bearing — it cross-checks env var names, CLI flag names, error codes, and a few specific value drifts across `backpropagate/**/*.py`, the handbook (`site/src/content/docs/handbook/*.md`), and `llms.txt`. If it fires, the message names which surface is out of sync. Per the v1.3 [[grep-all-instances-when-fixing-pattern]] doctrine, when you fix one drift instance, grep the rest of the repo for siblings — Wave 3.5 found 4 sibling drift sites across 5 handbook files via this approach.

The drift gate also runs as a `pre-push` pre-commit hook (CIDOCS-F-010, v1.4 Wave 6a) — `pre-commit install` wires it up so `git push` fires the check locally before CI does. v1.4 Wave 6a extended the gate to a 5-class scanner (argparse defaults vs handbook copy; llms.txt env-var names vs runtime reads; env-vars.md defaults vs `config.py` source-of-truth; error-codes.md "Fix" column cross-references; CI workflow severity claims vs flag semantics) so the next within-swarm doc-lie gets caught at commit time rather than in Wave N+1 audit.

## Code Style

We use the following tools to maintain code quality:

- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Pre-commit** hooks for automated checks

### Running Checks

```bash
# Lint with ruff
ruff check backpropagate/

# Format with ruff
ruff format backpropagate/

# Type check with mypy
mypy backpropagate/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backpropagate --cov-report=html

# Run specific test file
pytest tests/test_slao.py -v

# Run tests matching a pattern
pytest -k "merge" -v

# Skip slow tests
pytest -m "not slow"
```

### Mutation Testing

We use mutation testing to validate test quality:

```bash
# Run mutation tests (WSL recommended on Windows)
mutmut run --paths-to-mutate backpropagate/slao.py

# View results
mutmut results

# Show specific mutant
mutmut show 42
```

Target: 70%+ kill rate indicates good test coverage.

### Property-Based Testing

We use Hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=100))
def test_scale_factor_positive(run_index):
    scale = 1 / math.sqrt(run_index)
    assert scale > 0
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Write tests** for new functionality
4. **Ensure all tests pass**: `pytest`
5. **Run linting**: `ruff check backpropagate/`
6. **Commit** with a clear message
7. **Push** to your fork
8. **Open a Pull Request**

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add perplexity-based dataset filtering
fix: handle empty LoRA state in merge
docs: update export API examples
test: add property tests for SLAO merger
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding/updating tests
- `refactor:` - Code change that neither fixes nor adds
- `perf:` - Performance improvement
- `chore:` - Maintenance tasks

## Architecture Guidelines

### Design Principles

1. **Modular by default** - Features should be optional extras
2. **Smart defaults** - Works out of the box without configuration
3. **Windows-first** - No multiprocessing nightmares
4. **Fail gracefully** - Helpful error messages
5. **Type-safe** - Full type hints

### Adding New Features

1. Consider if it should be an optional extra
2. Add to `pyproject.toml` under `[project.optional-dependencies]`
3. Use lazy imports in `__init__.py`
4. Add feature flag detection in `feature_flags.py`
5. Write tests (aim for 80%+ coverage on new code)
6. Update README.md with usage examples

### Security Considerations

- Never use `pickle` for model loading (use `safetensors`)
- Validate all user inputs
- Use `weights_only=True` for `torch.load()`
- Check for path traversal in file operations
- See [`SECURITY.md`](SECURITY.md) for the reporting policy and supported versions; the operator-facing threat model + auth-middleware mode matrix lives at [`site/src/content/docs/handbook/security.md`](site/src/content/docs/handbook/security.md) (rendered at the handbook's `/handbook/security/` page). The historical `SECURITY_AUDIT_REPORT.md` is a stub kept for backlink compatibility.

## Issue Reporting

### Bug Reports

Include:
- **`run_id`** — printed at startup as `run_started run_id=<uuid>` and exposed as `TrainingRun.run_id` / `RunResult.run_id`. This correlates every log line, checkpoint, and SLAO merge for the failing run; it is the single most useful thing in a bug report.
- **The structured error code** — the `[CODE_NAME]: message` line in stderr. See the full catalog at `site/src/content/docs/handbook/error-codes.md` or the rendered handbook page.
- Python version
- PyTorch version
- GPU model and VRAM (a copy-paste of `backprop info` covers all four)
- Operating system
- Minimal reproduction code
- **Error traceback** — stderr in non-verbose mode is automatically redacted (Bearer tokens, `sk-*`, `hf_*`, AWS access keys, `password=`/`token=`/`api_key=` pairs are scrubbed), so it is **safe to paste** as-is. For the full unredacted output, re-run with `--verbose` and review the output for any secrets before posting.

### Feature Requests

- Describe the use case
- Propose an API design
- Consider backward compatibility

## Repo labels

The CI failure-observability path (`.github/workflows/nightly-train-smoke.yml`, `.github/workflows/post-publish-smoke.yml`, `.github/workflows/mutmut.yml`) opens GitHub Issues with labels when a smoke fails or a mutmut baseline rolls. GitHub requires those labels to exist on the repo before `gh issue create --label X` will apply them — otherwise the command silently no-ops and the issue is never created.

To keep the path self-healing on a fresh repo, freshly-renamed label, or after a label is accidentally deleted, each issue-creation workflow runs `gh label create <name> --force` idempotently just before the `gh issue create` step. The labels currently auto-bootstrapped are:

| Label | Used by | Color |
|-------|---------|-------|
| `ci` | All CI failure-observability workflows | `#0E8A16` (green) |
| `nightly-smoke` | `nightly-train-smoke.yml` | `#FBCA04` (yellow) |
| `post-publish-smoke` | `post-publish-smoke.yml` | `#FBCA04` (yellow) |
| `mutmut-baseline` | `mutmut.yml` baseline-refresh PRs | `#FBCA04` (yellow) |

If you want to retire one of these, drop the matching `gh label create` line from the bootstrap step in the workflow rather than relying on the label not existing on the repo.

## Dogfood-swarm rhythm

Backpropagate's larger releases are built as multi-wave parallel audits rather than one big PR. Each wave produces a JSON audit corpus (under `swarms/<release>/wave-<N>/`); the corpus seeds the next wave's amend work. The release cycle typically runs:

1. **Wave 1 — Stage A audit.** 5 parallel agents audit the codebase across backend / bridge / frontend / tests / ci-docs. Output: a JSON file per agent listing findings classified CRITICAL / HIGH / MEDIUM / LOW.
2. **Wave 2 — Stage A amend.** 5 parallel agents fix the CRITICAL / HIGH findings from Wave 1. Output: a PR with the closures + targeted regression tests.
3. **Wave 3 — Stage B audit.** Repeat audit against the Wave 2 amend to surface anything Wave 1 missed or that Wave 2 introduced. Wave 3.5 is a smaller follow-up amend when Stage B finds anything Stage-A-caliber.
4. **Wave 4 — Stage C humanization.** Cross-cutting copy / UX / docs pass. The audit is "does this read for the first-time operator?" not "is this technically correct?"
5. **Wave 5 — Feature audit.** Forward-looking — what new capabilities should land in this release.
6. **Wave 6 — Foundation features.** The actual feature build, gated on the audit closures from earlier waves.

If you're contributing during a wave, the coordinator will say so in the issue. Most contributions land outside the wave structure as one-off PRs — the swarm cadence is for the maintainer's coordination convenience, not a barrier to drive-by contributions.

## Questions?

- For "how do I do X" / "is this the right pattern" / "did anyone hit this before" — use [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). The Q&A category is the canonical channel; accepted answers help the next person searching. Ideas + Show and Tell categories cover roadmap and community-built recipes.
- For confirmed bugs — open an [issue](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml). The template requires `run_id` + error code + `backprop info` output.
- For security issues — file privately via the [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) form; see [SECURITY.md](SECURITY.md).
- Check existing issues + discussions first to avoid duplicates.
- Be respectful and constructive.

Thank you for contributing!
