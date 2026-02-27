#!/usr/bin/env bash
# verify.sh — single-command verification (Ship Gate D1)
set -euo pipefail

echo "=== Lint ==="
ruff check backpropagate/

echo "=== Type check ==="
mypy backpropagate/ --ignore-missing-imports

echo "=== Tests ==="
pytest tests/ -x --timeout=60

echo "=== Build ==="
python -m build

echo "=== Twine check ==="
twine check dist/*

echo "✓ verify passed"
