"""
Windowed BSS-eval metrics (SDR, SIR, SAR) built on mir_eval.

Drop-in replacement for museval.metrics.bss_eval that has zero system
dependencies (no ffmpeg / stempeg / musdb required).

Return convention mirrors museval so existing callers need no changes:
    sdr, sir, isr, sar, permutation = windowed_bss_eval(references, estimates, ...)

Shapes follow museval:
    references, estimates : (n_src, time, channels)
    sdr, sir, isr, sar    : (n_src, n_windows)  — float64 arrays
    permutation           : list[int]            — best permutation (0-indexed)
"""

from __future__ import annotations

import numpy as np


def _to_mono(x: np.ndarray) -> np.ndarray:
    """(time, channels) or (time,) → (time,) mono by averaging channels."""
    if x.ndim == 2:
        return x.mean(axis=-1)
    return x


def _bss_eval_window(ref_win: np.ndarray, est_win: np.ndarray) -> tuple:
    """
    Compute SDR/SIR/SAR for one window.

    Args:
        ref_win: (n_src, time) mono reference
        est_win: (n_src, time) mono estimate

    Returns:
        sdr, sir, sar : (n_src,) arrays
    """
    import mir_eval.separation as me_sep

    # mir_eval expects float64
    ref_win = ref_win.astype(np.float64)
    est_win = est_win.astype(np.float64)

    # Skip windows where any source is silent
    if np.any(np.sum(ref_win ** 2, axis=-1) < 1e-12):
        nan = np.full(ref_win.shape[0], np.nan)
        return nan, nan, nan

    try:
        sdr, sir, sar, _ = me_sep.bss_eval_sources(ref_win, est_win)
    except Exception:
        nan = np.full(ref_win.shape[0], np.nan)
        return nan, nan, nan

    return sdr, sir, sar


def windowed_bss_eval(
    references: np.ndarray,
    estimates: np.ndarray,
    window: int,
    hop: int,
    compute_permutation: bool = True,
    **_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Windowed BSS-eval: SDR, SIR, (ISR≈SIR), SAR, permutation.

    Mirrors the museval.metrics.bss_eval signature so existing callers
    can be updated with a one-line import swap.

    Args:
        references: (n_src, time, channels)  float array
        estimates:  (n_src, time, channels)  float array
        window:     window length in samples
        hop:        hop size in samples
        compute_permutation: if True, try each permutation and keep best
        **_kwargs:  ignored (for API compatibility)

    Returns:
        sdr : (n_src, n_windows)
        sir : (n_src, n_windows)
        isr : (n_src, n_windows)  — same as sir (ISR not fully supported)
        sar : (n_src, n_windows)
        permutation : list[int]  — best permutation indices
    """
    n_src, T, _ = references.shape

    # Convert to mono: (n_src, T)
    ref_mono = np.stack([_to_mono(references[i]) for i in range(n_src)])
    est_mono = np.stack([_to_mono(estimates[i]) for i in range(n_src)])

    # Build list of window start positions
    starts = list(range(0, max(1, T - window + 1), hop))
    n_win = len(starts)

    sdr_all = np.full((n_src, n_win), np.nan)
    sir_all = np.full((n_src, n_win), np.nan)
    sar_all = np.full((n_src, n_win), np.nan)

    for wi, start in enumerate(starts):
        end = min(start + window, T)
        ref_w = ref_mono[:, start:end]
        est_w = est_mono[:, start:end]
        sdr_w, sir_w, sar_w = _bss_eval_window(ref_w, est_w)
        sdr_all[:, wi] = sdr_w
        sir_all[:, wi] = sir_w
        sar_all[:, wi] = sar_w

    # Best permutation: the one that maximises median SDR across sources
    if compute_permutation and n_src > 1:
        import itertools

        best_perm = list(range(n_src))
        best_score = -np.inf
        for perm in itertools.permutations(range(n_src)):
            score = 0.0
            for src_i, est_j in enumerate(perm):
                ref_w = ref_mono[src_i]
                est_w = est_mono[est_j]
                # Re-evaluate under this permutation assignment
                vals = []
                for start in starts:
                    end = min(start + window, T)
                    rw = ref_w[start:end][np.newaxis]
                    ew = est_w[start:end][np.newaxis]
                    sdrs, _, _ = _bss_eval_window(rw, ew)
                    if np.isfinite(sdrs[0]):
                        vals.append(sdrs[0])
                if vals:
                    score += float(np.median(vals))
            if score > best_score:
                best_score = score
                best_perm = list(perm)

        # Re-order outputs to match best permutation
        sdr_all = sdr_all[best_perm]
        sir_all = sir_all[best_perm]
        sar_all = sar_all[best_perm]
    else:
        best_perm = list(range(n_src))

    isr_all = sir_all.copy()  # ISR not independently computed; approximate with SIR
    return sdr_all, sir_all, isr_all, sar_all, best_perm
