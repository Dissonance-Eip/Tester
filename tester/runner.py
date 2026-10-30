"""
Pipeline orchestration.

``run(original, perturbed)`` is the single public entry point — it loads
every model, runs every metric, and returns the assembled results.
The result schema lives in ``result``; side effects (console output, JSON
file) are delegated to ``report``.
"""

from typing import Any, NamedTuple

import torch

from . import models
from .audio import load
from .config import SNR_WINDOW_SECONDS
from .metrics import genre, lyrics, snr
from .metrics.genre import Genres
from .report import print_summary, save_json
from .result import RunResult, snr_label

DEFAULT_OUTPUT = "results.json"
_PREVIEW_CHARS = 120


class _LyricsStep(NamedTuple):
    original: str
    perturbed: str
    confusion: float


class _Audio(NamedTuple):
    """A file decoded once and shared by the SNR, MAEST and MERT steps."""

    path: str
    waveform: torch.Tensor  # [channels, samples]
    sr: int


class _GenreStep(NamedTuple):
    original: Genres
    perturbed: Genres
    encoder_confusion: float


def run(
    original_path: str, perturbed_path: str, output_path: str = DEFAULT_OUTPUT
) -> dict[str, Any]:
    """Evaluate a perturbation against the original audio.

    Runs three metrics:

    1. **SNR**             — how audible the perturbation is to humans
    2. **Lyrics confusion** — does Whisper hear different words?
    3. **Genre + encoder**  — does MAEST relabel? does MERT see a different track?

    All metrics use identical settings on both files so the comparison
    is fair. Each file is decoded once and the tensors feed SNR, MAEST
    and MERT; Whisper decodes the file itself (faster-whisper's own
    decoder and resampler, which its transcripts depend on).

    Args:
        original_path:  Path to the clean reference audio.
        perturbed_path: Path to the poisoned audio.
        output_path:    Where to write the JSON report.

    Returns:
        A structured results dictionary. Also writes it to ``output_path``
        and prints a summary to stdout as side effects.
    """
    print("\nLoading models...")
    models.load_all()
    print(f"Evaluating poison: {original_path}  →  {perturbed_path}\n")

    original = _Audio(original_path, *load(original_path))
    perturbed = _Audio(perturbed_path, *load(perturbed_path))

    snr_result = _snr_step(original, perturbed)
    lyrics_result = _lyrics_step(original_path, perturbed_path)
    genre_result = _genre_step(original, perturbed)

    results = RunResult(
        original_path=original_path,
        perturbed_path=perturbed_path,
        snr=snr_result,
        original_lyrics=lyrics_result.original,
        perturbed_lyrics=lyrics_result.perturbed,
        lyrics_confusion=lyrics_result.confusion,
        original_genres=genre_result.original,
        perturbed_genres=genre_result.perturbed,
        encoder_confusion=genre_result.encoder_confusion,
    ).to_dict()

    print_summary(results)
    save_json(results, output_path)
    return results


# ── Steps ────────────────────────────────────────────────────
def _snr_step(original: _Audio, perturbed: _Audio) -> snr.SnrResult:
    print("── Step 1/3: Human perceptual quality (SNR) ────")
    result = snr.compute_waveforms(
        original.waveform, original.sr, perturbed.waveform, perturbed.sr
    )
    print(f"  SNR: {result.snr_db:.1f} dB over the first {SNR_WINDOW_SECONDS} s")
    print(f"  Segmental SNR: min {result.segmental_min_db:.1f} dB, "
          f"mean {result.segmental_mean_db:.1f} dB  ({snr_label(result.segmental_min_db)})")
    if result.original_silent:
        print("  Original is silent in the window: SNR is undefined, counted as a fail.")
    print()
    return result


def _lyrics_step(original_path: str, perturbed_path: str) -> _LyricsStep:
    print("── Step 2/3: AI lyrics confusion (Whisper) ─────")
    original = lyrics.transcribe(original_path)
    perturbed = lyrics.transcribe(perturbed_path)
    confusion = lyrics.confusion(original, perturbed)
    print(f"  Original:  {original[:_PREVIEW_CHARS]}")
    print(f"  Perturbed: {perturbed[:_PREVIEW_CHARS]}")
    print(f"  Confusion: {confusion:.2%}\n")
    return _LyricsStep(original, perturbed, confusion)


def _genre_step(original: _Audio, perturbed: _Audio) -> _GenreStep:
    print("── Step 3/3: AI genre classification + encoder ─")
    print("  Classifying original (MAEST)...")
    original_genres = genre.classify_waveform(original.waveform, original.sr)
    print("  Classifying perturbed (MAEST)...")
    perturbed_genres = genre.classify_waveform(perturbed.waveform, perturbed.sr)
    print("  Original  genres:", _format_genres(original_genres))
    print("  Perturbed genres:", _format_genres(perturbed_genres))

    print("\n  Computing encoder confusion (MERT)...")
    confusion = genre.confusion(
        genre.fingerprint_waveform(original.waveform, original.sr),
        genre.fingerprint_waveform(perturbed.waveform, perturbed.sr),
    )
    print(f"  Encoder confusion: {confusion:.2%}\n")
    return _GenreStep(original_genres, perturbed_genres, confusion)


def _format_genres(genres: Genres) -> str:
    """Pretty-print genre tuples for the console."""
    return "  ".join(f"{g} ({s:.0%})" for g, s in genres)
