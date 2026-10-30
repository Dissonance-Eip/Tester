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
from ..audio import load, resample, to_mono, trim
from ..config import DEVICE, MAEST_SR, MAX_AUDIO_SECONDS, MERT_SR


# ── MAEST: discrete genre labels ─────────────────────────────
def classify(audio_path: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Classify an audio file into Discogs genres.

    Args:
        audio_path: Path to the audio file.
        top_k:      Number of top labels to return.

    Returns:
        List of ``(label, probability)`` tuples sorted by probability
        descending. **Scores are independent sigmoid probabilities** —
        they do NOT sum to 1 (a song can legitimately match multiple
        Discogs tags).
    """
    maest_model = models.maest()

    waveform, sr = load(audio_path)
    waveform = to_mono(waveform)
    waveform = trim(waveform, sr, MAX_AUDIO_SECONDS)
    waveform = resample(waveform, sr, MAEST_SR)

    audio = waveform.unsqueeze(0)  # [1, T] — kept on CPU; see models.maest()

    with torch.no_grad():
        activations, labels = maest_model.predict_labels(audio)

    scored = zip(labels, activations.tolist(), strict=True)
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ── MERT: continuous music embeddings ────────────────────────
def fingerprint(audio_path: str) -> torch.Tensor:
    """Compute a deterministic music fingerprint with MERT.

    Averages every transformer hidden layer (not just the last) for a
    richer representation, then mean-pools over time to produce a fixed
    1-D vector.

    Args:
        audio_path: Path to the audio file.

    Returns:
        1-D float tensor on CPU representing the music's latent features.
        Same input always produces the same fingerprint.
    """
    mert_model, mert_extractor = models.mert()

    waveform, sr = load(audio_path)
    waveform = to_mono(waveform)
    waveform = trim(waveform, sr, MAX_AUDIO_SECONDS)
    waveform = resample(waveform, sr, MERT_SR)

    inputs = mert_extractor(
        waveform.numpy(), sampling_rate=MERT_SR, return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs   = mert_model(**inputs, output_hidden_states=True)
        hidden    = torch.stack(outputs.hidden_states)     # [L, 1, T, D]
        embedding = hidden.mean(dim=(0, 2)).squeeze(0)     # [D]

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
    return 1.0 - torch.nn.functional.cosine_similarity(
        fp1.unsqueeze(0), fp2.unsqueeze(0)
    ).item()
