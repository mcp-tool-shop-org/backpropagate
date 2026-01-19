# Contributing to Backpropagate

Thank you for your interest in contributing to Backpropagate! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for full testing)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/mikeyfrilot/backpropagate
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
- See `SECURITY_AUDIT_REPORT.md` for full guidelines

## Issue Reporting

### Bug Reports

Include:
- Python version
- PyTorch version
- GPU model and VRAM
- Operating system
- Minimal reproduction code
- Full error traceback

### Feature Requests

- Describe the use case
- Propose an API design
- Consider backward compatibility

## Questions?

- Open an issue for questions
- Check existing issues first
- Be respectful and constructive

Thank you for contributing!
