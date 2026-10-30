"""Shared fixtures: write tiny synthetic WAV files from numpy arrays."""

import numpy as np
import pytest
import soundfile as sf

from .signals import SAMPLE_RATE


@pytest.fixture
def write_wav(tmp_path):
    """Return a function that writes ``samples`` to a float WAV and returns its path."""

    def _write(name: str, samples: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        path = tmp_path / f"{name}.wav"
        sf.write(str(path), samples.astype(np.float32), sr, subtype="FLOAT")
        return str(path)

    return _write

