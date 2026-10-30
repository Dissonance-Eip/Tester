"""Shared MERT embedding and cosine distance (issue #4).

The real MERT model is replaced by a tiny fake so the tests need no download.
"""

from types import SimpleNamespace

import pytest
import torch

from tester import embedding
from tester.config import MERT_SR

NUM_LAYERS = 3
LAYER_WEIGHTS = torch.tensor([1.0, 2.0, 3.0])


class FakeMert:
    """Stands in for MERT: hidden layer ``l`` at time ``t`` is ``[w_l·x_t, -w_l·x_t]``."""

    def __init__(self):
        self.seen = None

    def __call__(self, input_values, output_hidden_states):
        assert output_hidden_states is True
        self.seen = input_values
        frames = input_values.unsqueeze(-1)  # [1, T, 1]
        layers = tuple(torch.cat([w * frames, -w * frames], dim=-1) for w in LAYER_WEIGHTS)
        return SimpleNamespace(hidden_states=layers)


@pytest.fixture
def fake_mert(monkeypatch):
    model = FakeMert()
    extractor = SimpleNamespace(do_normalize=False)
    monkeypatch.setattr(embedding.models, "mert", lambda: (model, extractor))
    monkeypatch.setattr(embedding, "DEVICE", "cpu")
    return model, extractor


# ── mert_embed ───────────────────────────────────────────────
def test_embedding_is_the_mean_over_layers_and_time(fake_mert):
    wav = torch.tensor([0.1, 0.2, 0.3, 0.6])

    emb = embedding.mert_embed(wav, MERT_SR)

    expected = LAYER_WEIGHTS.mean() * wav.mean()
    assert emb.shape == (2,)
    assert torch.allclose(emb, torch.stack([expected, -expected]))


def test_input_reaches_the_model_at_mert_rate(fake_mert):
    model, _ = fake_mert
    wav = torch.randn(MERT_SR // 2, generator=torch.Generator().manual_seed(0))

    embedding.mert_embed(wav, MERT_SR // 2)

    assert model.seen.shape == (1, MERT_SR)


def test_input_at_mert_rate_is_passed_through_unchanged(fake_mert):
    model, _ = fake_mert
    wav = torch.randn(1000, generator=torch.Generator().manual_seed(0))

    embedding.mert_embed(wav, MERT_SR)

    assert model.seen[0].equal(wav)


def test_normalises_when_the_extractor_asks_for_it(fake_mert):
    model, extractor = fake_mert
    extractor.do_normalize = True
    wav = torch.tensor([1.0, 2.0, 3.0, 4.0])

    embedding.mert_embed(wav, MERT_SR)

    seen = model.seen[0]
    assert seen.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert seen.var(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)


def test_gradient_flows_back_to_the_waveform(fake_mert):
    wav = torch.tensor([0.1, 0.2, 0.3, 0.6], requires_grad=True)

    embedding.mert_embed(wav, MERT_SR)[0].backward()

    assert wav.grad is not None
    assert torch.all(wav.grad != 0)


# ── cosine_distance ──────────────────────────────────────────
def test_identical_vectors_have_zero_distance():
    a = torch.tensor([1.0, 2.0, 3.0])

    assert embedding.cosine_distance(a, a).item() == pytest.approx(0.0, abs=1e-6)


def test_orthogonal_vectors_have_distance_one():
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 5.0])

    assert embedding.cosine_distance(a, b).item() == pytest.approx(1.0)


def test_opposite_vectors_have_distance_two():
    a = torch.tensor([1.0, -2.0, 0.5])

    assert embedding.cosine_distance(a, -3 * a).item() == pytest.approx(2.0)


def test_distance_is_a_differentiable_scalar():
    a = torch.tensor([1.0, 0.5], requires_grad=True)
    b = torch.tensor([0.0, 1.0])

    distance = embedding.cosine_distance(a, b)
    distance.backward()

    assert distance.dim() == 0
    assert a.grad is not None and torch.any(a.grad != 0)


def test_genre_confusion_uses_the_shared_distance():
    from tester.metrics import genre

    a, b = torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])

    assert genre.confusion(a, b) == embedding.cosine_distance(a, b).item()
