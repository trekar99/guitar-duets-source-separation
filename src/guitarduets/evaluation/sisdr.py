from __future__ import annotations

import itertools

import numpy as np


def si_sdr(estimated_sources, reference_sources, window: int, hop: int, compute_permutation: bool = False):
    num_sources = len(estimated_sources)
    num_windows = (estimated_sources[0].shape[0] - window) // hop + 1
    values = np.zeros((num_sources, num_windows))

    for idx in range(num_windows):
        start = idx * hop
        end = start + window
        estimated_window = np.stack([source[start:end, :] for source in estimated_sources], axis=1)
        reference_window = np.stack([source[start:end, :] for source in reference_sources], axis=1)

        for source_idx in range(num_sources):
            target = reference_window[:, :, source_idx]
            estimate = estimated_window[:, :, source_idx]
            target_norm = np.linalg.norm(target - np.mean(target))
            projection = np.sum(target * estimate) * target / target_norm**2
            distortion = estimate - projection
            numerator = np.sum(projection * projection)
            denominator = np.sum(distortion * distortion)
            values[source_idx, idx] = 10 * np.log10(numerator / denominator)

    if compute_permutation:
        best_perm = max(itertools.permutations(range(num_sources)), key=lambda perm: values[perm, :].mean())
        values = values[best_perm, :]

    return values

