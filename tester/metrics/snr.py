"""
Signal-to-noise ratio — how audible the perturbation is to a human.

Treats the per-sample difference ``original - perturbed`` as noise and
compares its RMS energy to the RMS energy of the original signal:

    SNR (dB) = 20 · log10( RMS(signal) / RMS(noise) )

Higher SNR means the perturbation is *less* audible.
"""

import math

import torch

from ..audio import load, resample


def compute(original_path: str, perturbed_path: str) -> float:
    """Compute the SNR between two audio files.

    The perturbed file is resampled to match the original if their sample
    rates differ. If the channel counts differ, the perturbed file is
    mixed down and broadcast to match the original.

    Args:
        original_path:  Path to the original (clean) audio file.
        perturbed_path: Path to the perturbed audio file.

    Returns:
        SNR in decibels. ``float('inf')`` if the two files are identical
        (no noise → infinite SNR).
    """
    orig, sr_orig = load(original_path)
    pert, sr_pert = load(perturbed_path)

    # Align sample rate
    pert = resample(pert, sr_pert, sr_orig)

    # Align length
    min_len = min(orig.shape[-1], pert.shape[-1])
    orig = orig[..., :min_len]
    pert = pert[..., :min_len]

    # Align channel count (mono-down + broadcast)
    if orig.shape[0] != pert.shape[0]:
        pert = pert.mean(0, keepdim=True).expand_as(orig)

    noise     = orig.float() - pert.float()
    rms_orig  = orig.float().pow(2).mean().sqrt().item()
    rms_noise = noise.pow(2).mean().sqrt().item()

    if rms_noise < 1e-10:
        return float("inf")

    return 20.0 * math.log10(rms_orig / rms_noise)
