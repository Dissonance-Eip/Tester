"""Genre metrics: encoder confusion on tensors, and MAEST/MERT plumbing with fake models."""

import pytest
import torch

from tester.metrics import genre

from .signals import SAMPLE_RATE, sine

EMBED_DIM = 8


# ── confusion (pure) ─────────────────────────────────────────
def test_identical_fingerprints_give_zero():
    fp = torch.arange(1.0, EMBED_DIM + 1)

    assert genre.confusion(fp, fp.clone()) == pytest.approx(0.0, abs=1e-6)


def test_orthogonal_fingerprints_give_one():
    assert genre.confusion(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == pytest.approx(1.0)


def test_opposite_fingerprints_give_two():
    fp = torch.tensor([1.0, -2.0, 3.0])

    assert genre.confusion(fp, -fp) == pytest.approx(2.0)


def test_confusion_ignores_scale():
    fp = torch.tensor([0.3, 0.1, -0.7])

    assert genre.confusion(fp, 5 * fp) == pytest.approx(0.0, abs=1e-6)


# ── classify / fingerprint with fake models ──────────────────
class FakeMaest:
    def __init__(self):
        self.inputs = []

    def predict_labels(self, audio):
        self.inputs.append(audio)
        return torch.tensor([0.1, 0.7, 0.4]), ["Jazz---Bop", "Rock---Punk", "Pop---Indie"]


def test_classify_ranks_labels_by_probability(write_wav, monkeypatch):
    maest = FakeMaest()
    monkeypatch.setattr(genre.models, "maest", lambda: maest)

    ranked = genre.classify(write_wav("song", sine(1.0)), top_k=2)

    assert [label for label, _ in ranked] == ["Rock---Punk", "Pop---Indie"]
    assert ranked[0][1] == pytest.approx(0.7)
    # Mono, batched, resampled to MAEST's rate.
    assert maest.inputs[0].shape == (1, genre.MAEST_SR * 1)


def test_fingerprint_feeds_a_trimmed_mono_window_to_mert(monkeypatch):
    seen = []

    def fake_embed(waveform, sr):
        seen.append((waveform, sr))
        return torch.ones(EMBED_DIM)

    monkeypatch.setattr(genre, "mert_embed", fake_embed)
    monkeypatch.setattr(genre, "MAX_AUDIO_SECONDS", 1)
    stereo = torch.from_numpy(sine(2.0)).float().repeat(2, 1)  # [2, 2 s]

    fp = genre.fingerprint_waveform(stereo, SAMPLE_RATE)

    assert torch.equal(fp, torch.ones(EMBED_DIM))
    waveform, sr = seen[0]
    assert sr == SAMPLE_RATE
    assert waveform.shape == (SAMPLE_RATE,)  # mono, first second only


def test_path_and_tensor_entry_points_agree(write_wav, monkeypatch):
    from tester.audio import load

    monkeypatch.setattr(genre.models, "maest", FakeMaest)
    monkeypatch.setattr(genre, "mert_embed", lambda waveform, sr: waveform[:EMBED_DIM])
    path = write_wav("song", sine(1.0))

    assert genre.classify(path) == genre.classify_waveform(*load(path))
    assert torch.equal(genre.fingerprint(path), genre.fingerprint_waveform(*load(path)))
