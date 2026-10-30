"""
MERT embedding and cosine distance, shared by the metric and the attack.

``mert_embed`` is the single definition of "how MERT sees a waveform": the
Tester's encoder-confusion metric (``metrics/genre.py``) scores it under
``torch.no_grad()``, and ``attack/`` backpropagates through it. Both must use
this function so the attack optimises exactly what the Tester measures.
"""

import torch

from . import models
from .audio import resample
from .config import DEVICE, MERT_SR

# Matches Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm.
_NORMALISE_EPS = 1e-7


def mert_embed(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Differentiable MERT embedding of a mono waveform.

    Resamples to ``MERT_SR`` on the waveform's own device, moves it to
    ``DEVICE``, applies the feature extractor's normalisation if it asks
    for one (MERT-v1-330M's does not), then averages every hidden layer
    and mean-pools over time.

    Args:
        waveform: 1D mono waveform at ``sr``. Gradients flow back to it.
        sr:       Sample rate of ``waveform`` in Hz.

    Returns:
        1D embedding ``[D]`` on ``DEVICE``.
    """
    model, extractor = models.mert()
    x = resample(waveform, sr, MERT_SR).to(DEVICE)
    if getattr(extractor, "do_normalize", False):
        x = (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + _NORMALISE_EPS)
    outputs = model(input_values=x.unsqueeze(0), output_hidden_states=True)
    hidden = torch.stack(outputs.hidden_states)  # [L, 1, T, D]
    return hidden.mean(dim=(0, 2)).squeeze(0)    # [D]


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine distance between two 1D embeddings, as a differentiable scalar.

    Returns:
        A 0-dim tensor in ``[0, 2]``: 0 = same direction, 1 = orthogonal,
        2 = opposite.
    """
    similarity = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
    return 1.0 - similarity.squeeze(0)
