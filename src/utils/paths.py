from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

