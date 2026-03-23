from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import argparse



def make_histograms(metrics_file: str | Path):

    metrics_file = Path(metrics_file)

    if not metrics_file.exists():
        raise FileNotFoundError(metrics_file)

    # Infer run + checkpoint from artifact path
    checkpoint_name = metrics_file.parent.name
    run_name = metrics_file.parent.parent.name

    save_dir = Path("outputs") / "plots" / run_name / checkpoint_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_file) as f:
        metrics = json.load(f)

    sources = list(metrics.keys())
    metric_names = ["SDR", "SI-SDR", "SIR", "ISR", "SAR"]
    print(sources)

    for metric in metric_names:

        source_values = {}

        for source in sources:

            values = []

            for track_metrics in metrics[source].values():
                vals = track_metrics.get(metric)

                if vals is None:
                    continue

                values.extend(v for v in vals if np.isfinite(v))

            source_values[source] = values

        # Skip metric if empty
        if not any(len(v) for v in source_values.values()):
            continue

        all_vals = np.concatenate(
            [np.array(v) for v in source_values.values() if len(v) > 0]
        )

        min_val = int(np.floor(all_vals.min()))
        max_val = int(np.ceil(all_vals.max()))

        bin_edges = np.arange(min_val, max_val + 1, 1)

        fig, ax = plt.subplots()

        for source, values in source_values.items():
            if len(values) == 0:
                continue

            ax.hist(
                values,
                bins=bin_edges,
                alpha=0.5,
                label=source,
            )

        ax.set_xlabel(metric)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{metric} histogram")
        ax.legend()
        ax.grid(True)

        out_file = save_dir / f"{metric}_histogram.png"

        plt.savefig(out_file)
        plt.close()

        print(f"Saved {out_file}")

    print(f"\nHistograms saved in {save_dir}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file", required=True)
    args = parser.parse_args()

    make_histograms(args.metrics_file)
