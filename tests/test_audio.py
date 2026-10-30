"""Shared audio loader ``load_mono`` (issue #4)."""

import numpy as np
import pytest

from tester.audio import load_mono

from .signals import SAMPLE_RATE, sine


def test_stereo_is_averaged_to_mono(write_wav):
    left = sine(1.0, amp=0.5)
    right = sine(1.0, amp=0.1)
    path = write_wav("stereo", np.stack([left, right], axis=1))

    mono, sr = load_mono(path)

    assert sr == SAMPLE_RATE
    assert mono.dim() == 1
    np.testing.assert_allclose(mono.numpy(), (left + right) / 2, atol=1e-6)


def test_no_rate_and_no_seconds_keeps_the_whole_file_at_native_rate(write_wav):
    path = write_wav("clip", sine(1.5))

    mono, sr = load_mono(path)

    assert sr == SAMPLE_RATE
    assert mono.shape[-1] == int(1.5 * SAMPLE_RATE)


def test_seconds_trims_to_the_first_seconds(write_wav):
    path = write_wav("clip", sine(3.0))

    mono, sr = load_mono(path, seconds=2)

    assert sr == SAMPLE_RATE
    assert mono.shape[-1] == 2 * SAMPLE_RATE


def test_rate_resamples_after_trimming(write_wav):
    path = write_wav("clip", sine(3.0))

    mono, sr = load_mono(path, sr=SAMPLE_RATE * 2, seconds=1)

    assert sr == SAMPLE_RATE * 2
    assert mono.shape[-1] == 2 * SAMPLE_RATE


def test_trim_happens_at_the_native_rate(write_wav):
    """Trimming then resampling must equal the old load → mono → trim → resample chain."""
    from tester.audio import load, resample, to_mono, trim

    path = write_wav("clip", np.stack([sine(3.0), sine(3.0, freq=220.0)], axis=1))
    waveform, native = load(path)
    expected = resample(trim(to_mono(waveform), native, 2), native, 24_000)

    mono, sr = load_mono(path, sr=24_000, seconds=2)

    assert sr == 24_000
    assert mono.shape == expected.shape
    assert mono.equal(expected)


def test_file_shorter_than_seconds_is_returned_whole(write_wav):
    path = write_wav("short", sine(0.5))

    mono, _ = load_mono(path, seconds=2)

    assert mono.shape[-1] == pytest.approx(0.5 * SAMPLE_RATE)
