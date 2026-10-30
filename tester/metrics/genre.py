"""
Genre classification (MAEST) and encoder confusion (MERT).

Two independent signals:

* ``classify``    — MAEST gives multi-label sigmoid probabilities over the
  400-label Discogs genre/style taxonomy (e.g. ``Hip Hop---Trap``).
  These labels are *what a model would call the song* — useful for
  human-readable output.

* ``fingerprint`` + ``confusion`` — MERT produces a continuous embedding
  vector representing the music's latent features. Cosine distance
  between two embeddings tells you *how differently a music-understanding
  AI sees the two files*. This is the deeper signal for poisoning.

A successful poisoning attack should change BOTH:
the MAEST label *and* the MERT embedding.
"""

import torch

from .. import models
from ..audio import load, prepare_mono
from ..config import MAEST_SR, MAX_AUDIO_SECONDS
from ..embedding import cosine_distance, mert_embed

Genres = list[tuple[str, float]]


# ── MAEST: discrete genre labels ─────────────────────────────
def classify(audio_path: str, top_k: int = 5) -> Genres:
    """Classify an audio file into Discogs genres; see ``classify_waveform``."""
    return classify_waveform(*load(audio_path), top_k=top_k)


def classify_waveform(waveform: torch.Tensor, sr: int, top_k: int = 5) -> Genres:
    """Classify audio already in memory into Discogs genres.

    Args:
        waveform: ``[channels, samples]`` or ``[samples]`` tensor at ``sr``.
        sr:       Sample rate of ``waveform`` in Hz.
        top_k:    Number of top labels to return.

    Returns:
        List of ``(label, probability)`` tuples sorted by probability
        descending. **Scores are independent sigmoid probabilities** —
        they do NOT sum to 1 (a song can legitimately match multiple
        Discogs tags).
    """
    maest_model = models.maest()

    mono, _ = prepare_mono(waveform, sr, MAEST_SR, MAX_AUDIO_SECONDS)
    audio = mono.unsqueeze(0)  # [1, T] — kept on CPU; see models.maest()

    with torch.no_grad():
        activations, labels = maest_model.predict_labels(audio)

    scored = zip(labels, activations.tolist(), strict=True)
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ── MERT: continuous music embeddings ────────────────────────
def fingerprint(audio_path: str) -> torch.Tensor:
    """MERT fingerprint of an audio file; see ``fingerprint_waveform``."""
    return fingerprint_waveform(*load(audio_path))


def fingerprint_waveform(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Compute a deterministic music fingerprint with MERT.

    Averages every transformer hidden layer (not just the last) for a
    richer representation, then mean-pools over time to produce a fixed
    1-D vector. The embedding is ``embedding.mert_embed``, the same one
    ``attack/`` optimises against.

    Args:
        waveform: ``[channels, samples]`` or ``[samples]`` tensor at ``sr``.
        sr:       Sample rate of ``waveform`` in Hz.

    Returns:
        1-D float tensor on CPU representing the music's latent features.
        Same input always produces the same fingerprint.
    """
    mono, mono_sr = prepare_mono(waveform, sr, seconds=MAX_AUDIO_SECONDS)

    with torch.no_grad():
        embedding = mert_embed(mono, mono_sr)

    return embedding.cpu()


def confusion(fp1: torch.Tensor, fp2: torch.Tensor) -> float:
    """Cosine distance between two MERT fingerprints.

    Args:
        fp1: First fingerprint (any 1-D tensor).
        fp2: Second fingerprint (any 1-D tensor).

    Returns:
        Value in ``[0, 2]``. 0 = identical representation, 1 = orthogonal,
        2 = exact opposites. For our purposes anything above ~0.3 means
        a music-understanding model would see the two files as different.
    """
    return cosine_distance(fp1, fp2).item()
