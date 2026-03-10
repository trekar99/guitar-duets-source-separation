from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guitarduets.evaluation.summarize import summarize_results


def test_summarize_results_returns_medians():
    results = {
        "guitar1": {
            "track_a": {"SDR": [1, 3], "SIR": [1], "ISR": [2], "SAR": [4], "SI-SDR": [5]},
            "track_b": {"SDR": [2, 4], "SIR": [2], "ISR": [3], "SAR": [5], "SI-SDR": [6]},
        }
    }
    summary = summarize_results(results)
    assert summary["guitar1"]["SDR"] == 2.5
