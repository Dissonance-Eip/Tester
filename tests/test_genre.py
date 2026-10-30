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


class FakeMertOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class FakeMert:
    """Returns two hidden layers whose mean over layers and time is all ones."""

    def __call__(self, input_values, output_hidden_states):
        frames = 4
        return FakeMertOutput((
            torch.zeros(1, frames, EMBED_DIM),
            torch.full((1, frames, EMBED_DIM), 2.0),
        ))


def fake_extractor(waveform, sampling_rate, return_tensors):
    return {"input_values": torch.from_numpy(waveform).unsqueeze(0)}


def test_classify_ranks_labels_by_probability(write_wav, monkeypatch):
    maest = FakeMaest()
    monkeypatch.setattr(genre.models, "maest", lambda: maest)

    ranked = genre.classify(write_wav("song", sine(1.0)), top_k=2)

    assert [label for label, _ in ranked] == ["Rock---Punk", "Pop---Indie"]
    assert ranked[0][1] == pytest.approx(0.7)
    # Mono, batched, resampled to MAEST's rate.
    assert maest.inputs[0].shape == (1, genre.MAEST_SR * 1)


def test_fingerprint_averages_layers_and_time(write_wav, monkeypatch):
    monkeypatch.setattr(genre.models, "mert", lambda: (FakeMert(), fake_extractor))
    monkeypatch.setattr(genre, "DEVICE", "cpu")

    fp = genre.fingerprint(write_wav("song", sine(1.0, sr=SAMPLE_RATE)))

    assert fp.shape == (EMBED_DIM,)
    assert torch.allclose(fp, torch.ones(EMBED_DIM))
