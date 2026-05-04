"""
Additive white Gaussian noise (AWGN) for audio query clips.

By default the clip is RMS-normalized before noise is added, so a given
snr_db means the same ratio of signal power to noise power across clips
with different original loudness (snr_db = 10*log10(P_signal / P_noise)
with P = mean(x**2) on the normalized signal).
"""

from __future__ import annotations
import numpy as np


def signal_rms(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def normalize_rms(x: np.ndarray, target_rms: float = 1.0):
    """Scale waveform so sqrt(mean(x**2)) == target_rms."""
    x = np.asarray(x, dtype=np.float64)
    r = signal_rms(x)
    return (x * (target_rms / r)).astype(np.float64)


def add_awgn_gaussian(
    x: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    *,
    rms_normalize_first: bool = True,
    target_rms: float = 1.0,
    peak_limit: float | None = 1.0,
):
    """
    Add zero-mean Gaussian noise.

    If rms_normalize_first (default): scale clip to ``target_rms`` RMS, then
    draw noise with variance ``P_signal / 10**(snr_db/10)`` where
    P_signal = target_rms**2.

    If peak_limit is not None, scale the mixture down only if needed so
    max(abs(y)) <= peak_limit.

    If rms_normalize_first is False, uses the legacy rule: noise variance
    from the raw clip's mean power (loudness-dependent SNR).
    """
    x = np.asarray(x, dtype=np.float64)
    if rms_normalize_first:
        xn = normalize_rms(x, target_rms=target_rms)
        p_sig = float(target_rms**2)
    else:
        xn = x
        p_sig = float(np.mean(xn * xn) + 1e-12)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=xn.shape)
    y = xn + noise
    if peak_limit is not None:
        peak = float(np.max(np.abs(y)))
        if peak > peak_limit and peak > 1e-12:
            y = y * (peak_limit / peak)
    if not rms_normalize_first:
        y = np.clip(y, -1.2, 1.2)
    return y.astype(np.float32)
