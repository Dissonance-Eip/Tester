"""
Signal-to-noise ratio — how audible the perturbation is to a human.

Treats the per-sample difference ``original - perturbed`` as noise and
compares its RMS energy to the RMS energy of the original signal:

    SNR (dB) = 20 · log10( RMS(signal) / RMS(noise) )

Higher SNR means the perturbation is *less* audible.

Only the first ``SNR_WINDOW_SECONDS`` are measured: the same audio the
models see. Measuring the whole file would dilute a perturbation confined
to that window and overstate how quiet it is.

A single whole-window number can still hide a short, loud burst, so the
window is also cut into ``SNR_SEGMENT_SECONDS`` frames (segmental SNR).
The pass/fail gate uses the quietest frame (``segmental_min_db``).
"""

import math
from typing import List, NamedTuple

import torch

from ..audio import load, resample, trim
from ..config import SILENCE_RMS, SNR_SEGMENT_SECONDS, SNR_WINDOW_SECONDS

# Noise RMS below this is treated as no perturbation at all.
_NOISE_FREE_RMS = 1e-10


class SnrResult(NamedTuple):
    """SNR values in dB; ``inf`` = no perturbation, ``-inf`` = noise over silence."""

    snr_db: float             # whole model window
    segmental_min_db: float   # quietest-to-noise frame; used by the gate
    segmental_mean_db: float  # mean over frames that carry noise
    original_silent: bool     # original is silent over the whole window


def compute(original_path: str, perturbed_path: str) -> SnrResult:
    """Compute whole-window and segmental SNR between two audio files.

    The perturbed file is resampled to match the original if their sample
    rates differ. If the channel counts differ, the perturbed file is
    mixed down and broadcast to match the original. Both are trimmed to
    the model window.

    Args:
        original_path:  Path to the original (clean) audio file.
        perturbed_path: Path to the perturbed audio file.

    Returns:
        An ``SnrResult``. Identical files give ``inf`` everywhere. A silent
        original gives ``original_silent=True`` and ``-inf`` (any
        perturbation over silence is audible) or ``inf`` (none).
    """
    orig, sr = load(original_path)
    pert, sr_pert = load(perturbed_path)

    # Align sample rate
    pert = resample(pert, sr_pert, sr)

    # Align length
    min_len = min(orig.shape[-1], pert.shape[-1])
    orig = orig[..., :min_len]
    pert = pert[..., :min_len]

    # Align channel count (mono-down + broadcast)
    if orig.shape[0] != pert.shape[0]:
        pert = pert.mean(0, keepdim=True).expand_as(orig)

    # Measure only what the models see
    orig = trim(orig.float(), sr, SNR_WINDOW_SECONDS)
    noise = orig - trim(pert.float(), sr, SNR_WINDOW_SECONDS)

    rms_orig, rms_noise = _rms(orig), _rms(noise)
    if rms_orig < SILENCE_RMS:
        value = math.inf if rms_noise < _NOISE_FREE_RMS else -math.inf
        return SnrResult(value, value, value, original_silent=True)

    frames = _segmental_db(orig, noise, int(sr * SNR_SEGMENT_SECONDS))
    if not frames:
        return SnrResult(math.inf, math.inf, math.inf, original_silent=False)

    return SnrResult(
        snr_db=_db(rms_orig, rms_noise),
        segmental_min_db=min(frames),
        segmental_mean_db=sum(frames) / len(frames),
        original_silent=False,
    )


def _rms(x: torch.Tensor) -> float:
    return x.pow(2).mean().sqrt().item() if x.numel() else 0.0


def _db(rms_signal: float, rms_noise: float) -> float:
    if rms_noise < _NOISE_FREE_RMS:
        return math.inf
    return 20.0 * math.log10(rms_signal / rms_noise)


def _segmental_db(orig: torch.Tensor, noise: torch.Tensor, frame_len: int) -> List[float]:
    """Per-frame SNR in dB for every frame that carries noise.

    Noise-free frames are skipped (their SNR is infinite and would swamp
    the mean). Where the original is near-silent its level is floored at
    ``SILENCE_RMS``, so noise added to silence scores low instead of
    dividing by zero.
    """
    frame_len = max(frame_len, 1)
    frames = []
    for start in range(0, orig.shape[-1], frame_len):
        rms_noise = _rms(noise[..., start:start + frame_len])
        if rms_noise < _NOISE_FREE_RMS:
            continue
        rms_orig = max(_rms(orig[..., start:start + frame_len]), SILENCE_RMS)
        frames.append(_db(rms_orig, rms_noise))
    return frames
