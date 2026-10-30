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

Layers, each usable on its own:

* ``rms(x)``                          — RMS as a tensor (also used by ``attack/``)
* ``snr_db(original, perturbed)``     — one number from two equal-shape tensors
* ``align(original, sr, perturbed, sr_perturbed)`` — match rate, length, channels
* ``measure(original, perturbed, sr)`` — window + whole and segmental SNR
* ``compute_waveforms(original, sr, perturbed, sr_perturbed)`` — align, measure
* ``compute(original_path, perturbed_path)`` — load files, then the above
"""

import math
from typing import NamedTuple

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


def rms(x: torch.Tensor) -> torch.Tensor:
    """Root-mean-square of ``x`` as a 0-dim tensor in ``x``'s dtype.

    Kept as a tensor so ``attack/`` can stay on its device and in float32.
    """
    return x.pow(2).mean().sqrt()


def snr_db(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """SNR in dB of ``perturbed`` against ``original``.

    Args:
        original:  Clean signal, any shape.
        perturbed: Perturbed signal, same shape as ``original`` (use
            ``align`` first for audio that differs in rate, length or
            channels).

    Returns:
        ``inf`` when the two are identical. When ``original`` is silent
        (RMS below ``SILENCE_RMS``) the ratio is undefined: ``inf`` if there
        is no perturbation either, ``-inf`` if there is (any perturbation
        over silence is audible).

    Raises:
        ValueError: if the shapes differ.
    """
    if original.shape != perturbed.shape:
        raise ValueError(
            f"shape mismatch: original {tuple(original.shape)} "
            f"vs perturbed {tuple(perturbed.shape)}; align them first"
        )
    rms_orig = _rms(original.float())
    rms_noise = _rms(original.float() - perturbed.float())
    if rms_orig < SILENCE_RMS:
        return math.inf if rms_noise < _NOISE_FREE_RMS else -math.inf
    return _db(rms_orig, rms_noise)


def align(
    original: torch.Tensor,
    sr: int,
    perturbed: torch.Tensor,
    sr_perturbed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bring ``perturbed`` to the rate, length and channel count of ``original``.

    The perturbed signal is resampled to ``sr``, both are cut to the shorter
    length, and if the channel counts differ the perturbed signal is mixed
    down to mono and broadcast to the original's channels.

    Args:
        original:     ``[channels, samples]`` tensor at ``sr``.
        sr:           Sample rate of ``original`` in Hz.
        perturbed:    ``[channels, samples]`` tensor at ``sr_perturbed``.
        sr_perturbed: Sample rate of ``perturbed`` in Hz.

    Returns:
        ``(original, perturbed)`` as float tensors of equal shape at ``sr``.
    """
    perturbed = resample(perturbed, sr_perturbed, sr)

    length = min(original.shape[-1], perturbed.shape[-1])
    original = original[..., :length].float()
    perturbed = perturbed[..., :length].float()

    if original.shape[0] != perturbed.shape[0]:
        perturbed = perturbed.mean(0, keepdim=True).expand_as(original)
    return original, perturbed


def measure(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    sr: int,
    window_seconds: int | None = None,
) -> SnrResult:
    """Whole-window and segmental SNR of two aligned signals.

    Args:
        original:       Clean signal, ``[channels, samples]`` at ``sr``.
        perturbed:      Perturbed signal, same shape as ``original``.
        sr:             Sample rate in Hz.
        window_seconds: Seconds measured from the start; defaults to
            ``SNR_WINDOW_SECONDS`` (the longest model window).

    Returns:
        An ``SnrResult``. Identical signals give ``inf`` everywhere. A silent
        original gives ``original_silent=True`` and ``-inf`` (any
        perturbation over silence is audible) or ``inf`` (none).
    """
    if window_seconds is None:
        window_seconds = SNR_WINDOW_SECONDS
    original = trim(original.float(), sr, window_seconds)
    perturbed = trim(perturbed.float(), sr, window_seconds)

    whole = snr_db(original, perturbed)
    if _rms(original) < SILENCE_RMS:
        return SnrResult(whole, whole, whole, original_silent=True)

    frames = _segmental_db(original, original - perturbed, int(sr * SNR_SEGMENT_SECONDS))
    if not frames:
        return SnrResult(math.inf, math.inf, math.inf, original_silent=False)

    return SnrResult(
        snr_db=whole,
        segmental_min_db=min(frames),
        segmental_mean_db=sum(frames) / len(frames),
        original_silent=False,
    )


def compute(original_path: str, perturbed_path: str) -> SnrResult:
    """Compute whole-window and segmental SNR between two audio files.

    Loads both files, ``align``\\ s the perturbed one to the original and
    ``measure``\\ s the first ``SNR_WINDOW_SECONDS``.

    Args:
        original_path:  Path to the original (clean) audio file.
        perturbed_path: Path to the perturbed audio file.

    Returns:
        An ``SnrResult``; see ``measure``.
    """
    return compute_waveforms(*load(original_path), *load(perturbed_path))


def compute_waveforms(
    original: torch.Tensor, sr: int, perturbed: torch.Tensor, sr_perturbed: int
) -> SnrResult:
    """``compute`` for audio already in memory: ``align`` then ``measure``.

    Args:
        original:     ``[channels, samples]`` tensor at ``sr``.
        sr:           Sample rate of ``original`` in Hz.
        perturbed:    ``[channels, samples]`` tensor at ``sr_perturbed``.
        sr_perturbed: Sample rate of ``perturbed`` in Hz.
    """
    original, perturbed = align(original, sr, perturbed, sr_perturbed)
    return measure(original, perturbed, sr)


def _rms(x: torch.Tensor) -> float:
    return rms(x).item() if x.numel() else 0.0


def _db(rms_signal: float, rms_noise: float) -> float:
    if rms_noise < _NOISE_FREE_RMS:
        return math.inf
    return 20.0 * math.log10(rms_signal / rms_noise)


def _segmental_db(orig: torch.Tensor, noise: torch.Tensor, frame_len: int) -> list[float]:
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
