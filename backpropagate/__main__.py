"""
Backpropagate - CLI Entry Point
Run with: python -m backpropagate

This module delegates to the main CLI module (cli.py).
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
