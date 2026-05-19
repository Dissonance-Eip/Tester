"""
Lyrics-confusion metric via Whisper.

Transcribes both audio files with identical settings and computes a
word-overlap-based confusion score:

    confusion = 1 - |overlap(words_orig, words_pert)| / max(|orig|, |pert|)

Higher confusion = Whisper heard meaningfully different words on the
perturbed file (i.e. the poisoning is working).

Whisper settings here are tuned for **speed and fairness**, not maximum
accuracy. The same settings are applied to both files, so the relative
comparison is meaningful even if absolute transcription quality is lower.
"""

from typing import Set

from .. import models
from ..config import WHISPER_MAX_SECONDS


def transcribe(audio_path: str) -> str:
    """Transcribe audio to a single space-joined string.

    Uses greedy decoding (``beam_size=1``), VAD filtering (skips long
    instrumental sections), and only the first ``WHISPER_MAX_SECONDS``
    of the file — typically the first verse and chorus.
    """
    model = models.whisper()
    segments, _ = model.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=True,
        clip_timestamps=[0, WHISPER_MAX_SECONDS],
    )
    return " ".join(seg.text.strip() for seg in segments)


def confusion(original_text: str, perturbed_text: str) -> float:
    """Word-overlap confusion score in ``[0, 1]``.

    Args:
        original_text:  Transcription of the original audio.
        perturbed_text: Transcription of the perturbed audio.

    Returns:
        0.0 when transcriptions share every word, 1.0 when they share
        none. Symmetric in its two arguments.
    """
    words_o: Set[str] = set(original_text.lower().split())
    words_p: Set[str] = set(perturbed_text.lower().split())

    if not words_o and not words_p:
        return 0.0
    if not words_o or not words_p:
        return 1.0

    overlap = words_o & words_p
    return 1.0 - len(overlap) / max(len(words_o), len(words_p))
