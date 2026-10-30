"""Tensor-level SNR API: snr_db, align and measure need no files or models (issue #3)."""

import math

import numpy as np
import pytest
import torch

from tester.metrics import snr

from .signals import SAMPLE_RATE, noise_at_snr, sine

TOLERANCE_DB = 0.5


def _tensor(samples: np.ndarray) -> torch.Tensor:
    """``[1, samples]`` float tensor, the layout ``tester.audio.load`` returns."""
    return torch.from_numpy(samples).float().unsqueeze(0)


# ── snr_db ───────────────────────────────────────────────────
def test_identical_input_gives_inf():
    clean = _tensor(sine(1.0))

    assert snr.snr_db(clean, clean.clone()) == math.inf


@pytest.mark.parametrize("target_db", [10.0, 20.0, 40.0])
def test_known_noise_gives_expected_db(target_db):
    clean = sine(1.0)
    noisy = clean + noise_at_snr(clean, target_db)

    assert snr.snr_db(_tensor(clean), _tensor(noisy)) == pytest.approx(target_db, abs=TOLERANCE_DB)


def test_scaling_both_signals_keeps_the_snr():
    clean = sine(1.0)
    noisy = clean + noise_at_snr(clean, 20.0)

    full = snr.snr_db(_tensor(clean), _tensor(noisy))
    quiet = snr.snr_db(_tensor(clean * 0.1), _tensor(noisy * 0.1))

    assert quiet == pytest.approx(full, abs=1e-3)


def test_silent_original_with_noise_is_minus_inf():
    silence = torch.zeros(1, SAMPLE_RATE)
    noise = _tensor(noise_at_snr(sine(1.0), 30.0))

    assert snr.snr_db(silence, noise) == -math.inf


def test_silent_original_and_silent_perturbed_is_inf():
    silence = torch.zeros(1, SAMPLE_RATE)

    assert snr.snr_db(silence, silence.clone()) == math.inf


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        snr.snr_db(torch.zeros(2, 100), torch.zeros(1, 100))


# ── align ────────────────────────────────────────────────────
def test_align_cuts_to_the_shorter_length():
    orig, pert = snr.align(_tensor(sine(2.0)), SAMPLE_RATE, _tensor(sine(1.0)), SAMPLE_RATE)

    assert orig.shape == pert.shape == (1, SAMPLE_RATE)


def test_align_mixes_down_a_channel_mismatch():
    mono = _tensor(sine(1.0))
    stereo = torch.cat([mono, mono])

    orig, pert = snr.align(stereo, SAMPLE_RATE, mono, SAMPLE_RATE)

    assert pert.shape == orig.shape == (2, SAMPLE_RATE)
    assert snr.snr_db(orig, pert) == math.inf


def test_align_resamples_a_rate_mismatch():
    clean_8k = sine(1.0, sr=SAMPLE_RATE)
    clean_16k = sine(1.0, sr=2 * SAMPLE_RATE)

    orig, pert = snr.align(_tensor(clean_8k), SAMPLE_RATE, _tensor(clean_16k), 2 * SAMPLE_RATE)

    assert orig.shape == pert.shape == (1, SAMPLE_RATE)
    # Same tone at another rate: only resampling error remains, far above 30 dB.
    assert snr.snr_db(orig, pert) > 30.0


def test_unaligned_rate_mismatch_scores_badly():
    # Without align, a 16 kHz copy against the 8 kHz original is a different signal.
    clean_8k = _tensor(sine(1.0, sr=SAMPLE_RATE))
    clean_16k = _tensor(sine(1.0, sr=2 * SAMPLE_RATE))[..., :SAMPLE_RATE]

    assert snr.snr_db(clean_8k, clean_16k) < 10.0


# ── measure ──────────────────────────────────────────────────
def test_measure_only_covers_the_window():
    clean = sine(4.0)
    noisy = clean.copy()
    noisy[:SAMPLE_RATE] += noise_at_snr(clean[:SAMPLE_RATE], 20.0)

    result = snr.measure(_tensor(clean), _tensor(noisy), SAMPLE_RATE, window_seconds=1)

    assert result.snr_db == pytest.approx(20.0, abs=TOLERANCE_DB)
    assert result.original_silent is False


def test_measure_segmental_min_catches_a_burst():
    clean = sine(4.0)
    noisy = clean.copy()
    burst = slice(SAMPLE_RATE, 2 * SAMPLE_RATE)
    noisy[burst] += noise_at_snr(clean[burst], 10.0)

    result = snr.measure(_tensor(clean), _tensor(noisy), SAMPLE_RATE, window_seconds=4)

    assert result.segmental_min_db == pytest.approx(10.0, abs=TOLERANCE_DB)
    assert result.segmental_min_db < result.snr_db


def test_measure_identical_input_is_inf_everywhere():
    clean = _tensor(sine(1.0))

    result = snr.measure(clean, clean.clone(), SAMPLE_RATE, window_seconds=1)

    assert result == snr.SnrResult(math.inf, math.inf, math.inf, original_silent=False)


def test_measure_silent_original_is_flagged():
    silence = torch.zeros(1, SAMPLE_RATE)
    noise = _tensor(noise_at_snr(sine(1.0), 30.0))

    result = snr.measure(silence, noise, SAMPLE_RATE, window_seconds=1)

    assert result == snr.SnrResult(-math.inf, -math.inf, -math.inf, original_silent=True)


# ── compute (files) with mismatched inputs ───────────────────
def test_compute_handles_rate_and_channel_mismatch(write_wav, monkeypatch):
    monkeypatch.setattr(snr, "SNR_WINDOW_SECONDS", 1)
    clean = sine(1.0)
    stereo_16k = np.stack([sine(1.0, sr=2 * SAMPLE_RATE)] * 2, axis=1)  # [samples, channels]

    result = snr.compute(
        write_wav("clean", clean),
        write_wav("stereo", stereo_16k, sr=2 * SAMPLE_RATE),
    )

    assert result.snr_db > 30.0
