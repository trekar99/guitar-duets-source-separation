from __future__ import annotations

from typing import Any

import math
import statistics


def _to_finite_list(value: Any) -> list[float]:
    """
    Normalize metric payloads into a flat list of finite floats.
    Accepts scalars or iterables.
    """
    if value is None:
        return []

    if isinstance(value, (int, float)):
        values = [float(value)]
    else:
        values = [float(v) for v in value]

    return [v for v in values if math.isfinite(v)]


def summarize_results(results: dict) -> dict:
    """
    Aggregate by taking a single global median over all finite frame values
    for each source and metric.

    This avoids the old median-of-medians behavior.
    """
    summary: dict[str, dict] = {}

    for source_name, tracks in results.items():
        source_summary: dict[str, Any] = {
            "num_tracks": len(tracks),
        }

        for metric_name in ("SDR", "SIR", "ISR", "SAR", "SI-SDR"):
            all_values: list[float] = []

            for track_metrics in tracks.values():
                all_values.extend(_to_finite_list(track_metrics.get(metric_name)))

            source_summary[metric_name] = (
                statistics.median(all_values) if all_values else None
            )

        summary[source_name] = source_summary

    return summary