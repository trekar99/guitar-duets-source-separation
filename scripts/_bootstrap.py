"""Resolve the repository root and ensure src/ is importable.

All scripts under scripts/ call ``bootstrap()`` at import time so that
``from src.…`` imports work regardless of the current working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = str(repo_root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return repo_root
