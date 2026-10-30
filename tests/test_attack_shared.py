"""``attack/`` uses the Tester's shared primitives and constants (issue #4)."""

import inspect

import numpy as np
import pytest
import torch

from attack import optimizer, run_uap, targets
from tester import embedding
from tester.config import MAX_AUDIO_SECONDS, SNR_GOOD

from .signals import SAMPLE_RATE, sine


def test_attack_defaults_to_the_tester_snr_threshold():
    for fn in (optimizer.attack, optimizer.attack_universal):
        assert inspect.signature(fn).parameters["target_snr_db"].default == SNR_GOOD


def test_cli_defaults_come_from_config(monkeypatch):
    from attack import run_attack

    seen = {}

    def fake_parse(self, *a, **kw):
        seen.update({act.dest: act.default for act in self._actions})
        raise SystemExit(0)

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", fake_parse)
    with pytest.raises(SystemExit):
        run_attack.main()
    assert seen["window"] == MAX_AUDIO_SECONDS
    assert seen["snr"] == SNR_GOOD

    seen.clear()
    with pytest.raises(SystemExit):
        run_uap.main()
    assert seen["seconds"] == MAX_AUDIO_SECONDS
    assert seen["snr"] == SNR_GOOD


def test_mert_target_embeds_with_the_shared_embedding(monkeypatch):
    calls = []

    def fake_embed(wav, sr):
        calls.append((wav, sr))
        return torch.zeros(4)

    monkeypatch.setattr(embedding.models, "mert", lambda: (object(), object()))
    monkeypatch.setattr(targets, "mert_embed", fake_embed)
    target = targets.MertTarget()
    wav = torch.ones(10)

    target.embed(wav, 44_100)

    assert calls == [(wav, 44_100)]


def test_uap_loads_every_song_at_one_rate(write_wav):
    train = [write_wav("a", sine(2.0)), write_wav("b", sine(1.5))]
    test = write_wav("c", sine(2.0))

    segs, test_seg, sr = run_uap.load_songs(train, test, seconds=1)

    assert sr == SAMPLE_RATE
    assert [s.shape[-1] for s in segs] == [SAMPLE_RATE, SAMPLE_RATE]
    assert test_seg.shape[-1] == SAMPLE_RATE


def test_uap_rejects_songs_with_mixed_sample_rates(write_wav):
    train = [write_wav("a", sine(2.0)), write_wav("b", sine(2.0, sr=16_000), sr=16_000)]
    test = write_wav("c", sine(2.0))

    with pytest.raises(ValueError, match="sample rate"):
        run_uap.load_songs(train, test, seconds=1)


def test_uap_rejects_a_test_song_at_another_rate(write_wav):
    train = [write_wav("a", sine(2.0)), write_wav("b", sine(2.0))]
    test = write_wav("c", sine(2.0, sr=16_000), sr=16_000)

    with pytest.raises(ValueError, match="sample rate"):
        run_uap.load_songs(train, test, seconds=1)


def test_uap_truncates_training_songs_to_the_shortest(write_wav):
    train = [write_wav("a", sine(2.0)), write_wav("b", sine(0.5))]
    test = write_wav("c", sine(2.0))

    segs, _, _ = run_uap.load_songs(train, test, seconds=1)

    assert {s.shape[-1] for s in segs} == {SAMPLE_RATE // 2}
    np.testing.assert_array_equal(segs[0].numpy(), sine(1.0)[: SAMPLE_RATE // 2].astype(np.float32))
