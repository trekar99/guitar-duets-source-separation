from __future__ import annotations

import statistics


def summarize_results(results: dict) -> dict:
    summary = {}
    for source_name, tracks in results.items():
        source_summary = {}
        for metric_name in ("SDR", "SIR", "ISR", "SAR", "SI-SDR"):
            values = [statistics.median(track_metrics[metric_name]) for track_metrics in tracks.values()]
            source_summary[metric_name] = statistics.median(values) if values else None
        summary[source_name] = source_summary
    return summary

