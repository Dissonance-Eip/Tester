"""SNR metric: model window, segmental SNR and silent input (issue #2)."""

import math

import numpy as np
import pytest

from tester.metrics import snr

from .signals import SAMPLE_RATE, noise_at_snr, sine

WINDOW_SECONDS = 1
TOLERANCE_DB = 0.5


@pytest.fixture(autouse=True)
def short_window(monkeypatch):
    """Use a 1 s model window so the test signals stay tiny."""
    monkeypatch.setattr(snr, "SNR_WINDOW_SECONDS", WINDOW_SECONDS, raising=False)


def test_identical_files_give_infinite_snr(write_wav):
    clean = sine(2.0)
    path = write_wav("clean", clean)

    result = snr.compute(path, path)

    assert result.snr_db == math.inf
    assert result.segmental_min_db == math.inf
    assert result.segmental_mean_db == math.inf


def test_known_noise_gives_expected_db(write_wav):
    clean = sine(1.0)
    noisy = clean + noise_at_snr(clean, 20.0)

    result = snr.compute(write_wav("clean", clean), write_wav("noisy", noisy))

    assert result.snr_db == pytest.approx(20.0, abs=TOLERANCE_DB)


def test_snr_only_covers_the_model_window(write_wav):
    # Noise only in the first second (the window the models see); the rest is clean.
    clean = sine(4.0)
    first = WINDOW_SECONDS * SAMPLE_RATE
    noisy = clean.copy()
    noisy[:first] += noise_at_snr(clean[:first], 20.0)

    result = snr.compute(write_wav("clean", clean), write_wav("noisy", noisy))

    # Whole-file SNR would read about 26 dB (noise diluted over 4 s).
    assert result.snr_db == pytest.approx(20.0, abs=TOLERANCE_DB)


def test_segmental_min_catches_a_localised_perturbation(write_wav, monkeypatch):
    monkeypatch.setattr(snr, "SNR_WINDOW_SECONDS", 4, raising=False)
    clean = sine(4.0)
    noisy = clean.copy()
    burst = slice(SAMPLE_RATE, 2 * SAMPLE_RATE)  # one loud second out of four
    noisy[burst] += noise_at_snr(clean[burst], 10.0)

    result = snr.compute(write_wav("clean", clean), write_wav("noisy", noisy))

    assert result.segmental_min_db == pytest.approx(10.0, abs=TOLERANCE_DB)
    assert result.segmental_min_db < result.snr_db
    assert result.segmental_min_db <= result.segmental_mean_db


def test_segmental_skips_noise_free_frames(write_wav, monkeypatch):
    monkeypatch.setattr(snr, "SNR_WINDOW_SECONDS", 4, raising=False)
    clean = sine(4.0)
    noisy = clean.copy()
    burst = slice(0, SAMPLE_RATE)
    noisy[burst] += noise_at_snr(clean[burst], 25.0)

    result = snr.compute(write_wav("clean", clean), write_wav("noisy", noisy))

    # Only the noisy frames count; untouched frames do not inflate the mean to inf.
    assert math.isfinite(result.segmental_mean_db)
    assert result.segmental_mean_db == pytest.approx(25.0, abs=TOLERANCE_DB)


def test_silent_original_with_perturbation_is_a_clear_fail(write_wav):
    silence = np.zeros(SAMPLE_RATE)
    perturbed = noise_at_snr(sine(1.0), 30.0)

    result = snr.compute(write_wav("silence", silence), write_wav("perturbed", perturbed))

    assert result.original_silent is True
    assert result.snr_db == -math.inf
    assert result.segmental_min_db == -math.inf


def test_silent_original_and_silent_perturbed_is_infinite(write_wav):
    path = write_wav("silence", np.zeros(SAMPLE_RATE))

    result = snr.compute(path, path)

    assert result.original_silent is True
    assert result.snr_db == math.inf


# ── Shared primitives (issue #4) ─────────────────────────────
def test_rms_of_a_constant_is_its_magnitude():
    import torch

    assert snr.rms(torch.full((100,), -0.25)).item() == pytest.approx(0.25)


def test_rms_returns_a_tensor_in_the_input_dtype():
    import torch

    value = snr.rms(torch.ones(10, dtype=torch.float32))

    assert isinstance(value, torch.Tensor)
    assert value.dtype == torch.float32


def test_snr_db_of_identical_signals_is_infinite():
    import torch

    signal = torch.from_numpy(sine(1.0)).float()

    assert snr.snr_db(signal, signal) == math.inf


def test_snr_db_of_known_noise():
    import torch

    clean = sine(1.0)
    noisy = clean + noise_at_snr(clean, 20.0)

    value = snr.snr_db(torch.from_numpy(clean).float(), torch.from_numpy(noisy).float())

    assert value == pytest.approx(20.0, abs=0.01)


def test_compute_whole_window_snr_matches_snr_db(write_wav):
    import torch

    clean = sine(1.0)
    noisy = clean + noise_at_snr(clean, 25.0)

    result = snr.compute(write_wav("clean", clean), write_wav("noisy", noisy))

    expected = snr.snr_db(torch.from_numpy(clean).float(), torch.from_numpy(noisy).float())
    assert result.snr_db == pytest.approx(expected, abs=1e-4)
