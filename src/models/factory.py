from __future__ import annotations

from src.models.demucs import Demucs
from src.models.hdemucs import HDemucs
from src.models.htdemucs import HTDemucs


def build_model(name: str, kwargs: dict | None = None):
    model_kwargs = kwargs or {}
    model_name = name.lower()
    if model_name == "demucs":
        return Demucs(**model_kwargs)
    if model_name == "hdemucs":
        return HDemucs(**model_kwargs)
    if model_name == "htdemucs":
        return HTDemucs(**model_kwargs)
    raise ValueError(f"Unsupported model: {name}")

