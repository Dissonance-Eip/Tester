"""
Audio I/O helpers.

Uses `soundfile` for loading so we don't depend on torchaudio's backend
system (which changes between PyTorch versions and currently requires
`torchcodec`). All functions are pure and stateless.
"""

from typing import Tuple

import torch
import torchaudio
import soundfile as sf


def load(path: str) -> Tuple[torch.Tensor, int]:
    """Load an audio file into a PyTorch tensor.

    Args:
        path: Path to a WAV / FLAC / OGG file.

    Returns:
        A tuple ``(waveform, sample_rate)`` where ``waveform`` has shape
        ``[channels, samples]`` and dtype ``float32``.
    """
    data, sr = sf.read(path, always_2d=True)   # [samples, channels]
    waveform = torch.from_numpy(data.T).float()  # [channels, samples]
    return waveform, sr


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Mix multichannel audio down to mono by averaging across channels.

    Args:
        waveform: Tensor of shape ``[channels, samples]`` or ``[samples]``.

    Returns:
        1D tensor of shape ``[samples]``.
    """
    if waveform.dim() == 1:
        return waveform
    return waveform.mean(dim=0)


def resample(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    """Resample audio to a target sample rate.

    No-op when the source and destination rates match.

    Args:
        waveform: Audio tensor (any shape; resampling is applied on last dim).
        src_sr: Original sample rate in Hz.
        dst_sr: Target sample rate in Hz.

    Returns:
        Resampled tensor.
    """
    if src_sr == dst_sr:
        return waveform
    return torchaudio.functional.resample(waveform, src_sr, dst_sr)


def trim(waveform: torch.Tensor, sr: int, seconds: int) -> torch.Tensor:
    """Trim audio to the first ``seconds`` seconds.

    Args:
        waveform: Audio tensor; trimming is applied on the last dim.
        sr: Sample rate of ``waveform``.
        seconds: Maximum desired length in seconds.

    Returns:
        Trimmed tensor (returns the original if it was already short enough).
    """
    return waveform[..., : sr * seconds]
