"""
Differentiable attack targets.

Each target wraps one of the Tester's models and exposes a *differentiable*
embedding of an audio waveform, so the optimizer can backprop a loss all the
way to the input samples.

Phase 0 ships only the MERT target. MERT is the model behind the Tester's
``genre_conf`` gate (``tester/metrics/genre.py`` fingerprint/confusion), and
it is the one model that is natively differentiable (HuggingFace PyTorch
module). The embedding is ``tester.embedding.mert_embed``, the very function
``genre.fingerprint`` scores, so the attack optimises what the Tester measures.
"""

import torch

from tester import models
from tester.embedding import mert_embed


class MertTarget:
    """Differentiable wrapper around the MERT music encoder."""

    def __init__(self, use_checkpointing: bool = False):
        # Reuse the Tester's lazy singleton — do not re-instantiate the model.
        self.model, _ = models.mert()

        # Optional gradient checkpointing — only useful if you hit a real memory
        # ceiling on very long windows. It trades speed for memory (recomputes
        # activations during backward), so it's OFF by default.
        if use_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
                print("  gradient checkpointing: enabled")
            except Exception as e:  # noqa: BLE001 — best-effort; falls back to normal
                print(f"  gradient checkpointing: unavailable ({e})")

    def embed(self, wav_native: torch.Tensor, native_sr: int) -> torch.Tensor:
        """Differentiable MERT embedding of a mono waveform.

        Args:
            wav_native: 1-D mono waveform at ``native_sr``, on ``DEVICE``,
                with ``requires_grad`` flowing from the perturbation.
            native_sr:  Sample rate of ``wav_native``.

        Returns:
            1-D embedding ``[D]``, the same one ``genre.fingerprint`` returns.
        """
        return mert_embed(wav_native, native_sr)

    @torch.no_grad()
    def clean_embed(self, wav_native: torch.Tensor, native_sr: int) -> torch.Tensor:
        """Fixed reference embedding of the unperturbed song (no gradient)."""
        return self.embed(wav_native, native_sr).detach()
