"""
Differentiable attack targets.

Each target wraps one of the Tester's models and exposes a *differentiable*
embedding of an audio waveform, so the optimizer can backprop a loss all the
way to the input samples.

Phase 0 ships only the MERT target. MERT is the model behind the Tester's
``genre_conf`` gate (``tester/metrics/genre.py`` fingerprint/confusion), and
it is the one model that is natively differentiable (HuggingFace PyTorch
module). We reproduce ``genre.fingerprint``'s embedding exactly — mean over all
hidden layers, pooled over time — but built from a tensor input so gradients
flow, instead of the numpy feature-extractor path the metric uses.
"""

import torch

from tester import models
from tester.audio import resample
from tester.config import DEVICE, MERT_SR


class MertTarget:
    """Differentiable wrapper around the MERT music encoder."""

    def __init__(self, use_checkpointing: bool = False):
        # Reuse the Tester's lazy singleton — do not re-instantiate the model.
        self.model, self.extractor = models.mert()
        # Probed offline: MERT's extractor has do_normalize=False, so the raw
        # waveform is fed straight in as input_values (no z-scoring needed).
        self._do_normalize = bool(getattr(self.extractor, "do_normalize", False))

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
            1-D embedding ``[D]`` matching ``genre.fingerprint``'s shape.
        """
        x = resample(wav_native, native_sr, MERT_SR)  # differentiable resample
        if self._do_normalize:
            x = (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-7)
        out = self.model(input_values=x.unsqueeze(0), output_hidden_states=True)
        hidden = torch.stack(out.hidden_states)  # [L, 1, T, D]
        return hidden.mean(dim=(0, 2)).squeeze(0)  # [D]

    @torch.no_grad()
    def clean_embed(self, wav_native: torch.Tensor, native_sr: int) -> torch.Tensor:
        """Fixed reference embedding of the unperturbed song (no gradient)."""
        return self.embed(wav_native, native_sr).detach()
