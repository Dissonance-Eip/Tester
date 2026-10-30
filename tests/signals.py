"""Small synthetic test signals."""

import numpy as np

SAMPLE_RATE = 8_000


def sine(seconds: float, sr: int = SAMPLE_RATE, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def noise_at_snr(signal: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """White noise scaled so that RMS(signal) / RMS(noise) equals ``snr_db``."""
    noise = np.random.default_rng(seed).standard_normal(signal.shape)
    sig_rms = np.sqrt(np.mean(signal ** 2))
    return noise * sig_rms / np.sqrt(np.mean(noise ** 2)) * 10 ** (-snr_db / 20)
