"""Utilities for dynamic imports and finding generator functions."""

from pathlib import Path

try:
    from rapidfuzz import process

    rapidfuzz = process
except ImportError:
    rapidfuzz = None


def module_path():
    return Path(__file__).parent.parent
