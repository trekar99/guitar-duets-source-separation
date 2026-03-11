from __future__ import annotations

import numpy as np


def _window_starts(num_samples: int, window: int, hop: int) -> list[int]:
    """
    Generate window start indices.

    If the signal is shorter than one full window, evaluate on the full signal once.
    If the final hop misses the tail, add one final window ending at the last sample.
    """
    if num_samples <= 0:
        return []

    if window <= 0 or hop <= 0:
        raise ValueError(f"window and hop must be positive, got window={window}, hop={hop}")

    if num_samples <= window:
        return [0]

    starts = list(range(0, num_samples - window + 1, hop))
    last_start = num_samples - window
    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def _si_sdr_one(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute SI-SDR for a single aligned estimate/target pair.

    Inputs may be mono or multi-channel; they are flattened after zero-meaning.
    Returns NaN for silent/near-silent targets.
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    target_zm = target - np.mean(target)
    estimate_zm = estimate - np.mean(estimate)

    target_vec = target_zm.reshape(-1)
    estimate_vec = estimate_zm.reshape(-1)

    target_energy = float(np.dot(target_vec, target_vec))
    if target_energy <= eps:
        return float("nan")

    alpha = float(np.dot(estimate_vec, target_vec)) / target_energy
    projection = alpha * target_vec
    distortion = estimate_vec - projection

    num = float(np.dot(projection, projection))
    den = float(np.dot(distortion, distortion))

    if num <= eps and den <= eps:
        return float("nan")

    return float(10.0 * np.log10((num + eps) / (den + eps)))


def si_sdr(
    estimated_sources: np.ndarray,
    reference_sources: np.ndarray,
    window: int,
    hop: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Framewise SI-SDR for already-aligned source estimates.

    Parameters
    ----------
    estimated_sources : np.ndarray
        Shape (time, sources, channels)
    reference_sources : np.ndarray
        Shape (time, sources, channels)
    window : int
        Window size in samples
    hop : int
        Hop size in samples
    eps : float
        Numerical stability constant

    Returns
    -------
    np.ndarray
        Shape (sources, num_windows)
    """
    estimates = np.asarray(estimated_sources, dtype=np.float64)
    references = np.asarray(reference_sources, dtype=np.float64)

    if estimates.ndim != 3 or references.ndim != 3:
        raise ValueError(
            "estimated_sources and reference_sources must have shape "
            "(time, sources, channels)"
        )

    if estimates.shape != references.shape:
        raise ValueError(
            f"Shape mismatch between estimates and references: "
            f"{estimates.shape} vs {references.shape}"
        )

    num_samples, num_sources, _ = estimates.shape
    starts = _window_starts(num_samples, window, hop)
    if not starts:
        raise ValueError("No valid windows could be constructed for SI-SDR.")

    values = np.full((num_sources, len(starts)), np.nan, dtype=np.float64)

    for w_idx, start in enumerate(starts):
        end = min(start + window, num_samples)
        est_win = estimates[start:end, :, :]
        ref_win = references[start:end, :, :]

        for source_idx in range(num_sources):
            values[source_idx, w_idx] = _si_sdr_one(
                estimate=est_win[:, source_idx, :],
                target=ref_win[:, source_idx, :],
                eps=eps,
            )

    return values