#!/bin/bash
# Mutation testing with mutmut
# Run this in WSL (Windows Subsystem for Linux)
#
# Usage:
#   ./scripts/run_mutmut.sh              # Full mutation test
#   ./scripts/run_mutmut.sh slao         # Test only slao.py
#   ./scripts/run_mutmut.sh results      # View results
#   ./scripts/run_mutmut.sh html         # Generate HTML report

set -e

cd "$(dirname "$0")/.."

case "${1:-run}" in
    run)
        echo "Running mutation tests (this takes a while)..."
        mutmut run
        ;;
    slao)
        echo "Running mutation tests on slao.py only..."
        mutmut run --paths-to-mutate backpropagate/slao.py
        ;;
    checkpoints)
        echo "Running mutation tests on checkpoints.py only..."
        mutmut run --paths-to-mutate backpropagate/checkpoints.py
        ;;
    results)
        echo "Mutation test results:"
        mutmut results
        ;;
    html)
        echo "Generating HTML report..."
        mutmut html
        echo "Report saved to html/"
        ;;
    browse)
        echo "Interactive browser (q to quit)..."
        mutmut browse
        ;;
    *)
        echo "Usage: $0 [run|slao|checkpoints|results|html|browse]"
        exit 1
        ;;
esac
