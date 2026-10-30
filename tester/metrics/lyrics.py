"""
Lyrics-confusion metric via Whisper.

Transcribes both audio files with identical settings, normalises the text
(case and punctuation) and computes a word-level edit distance:

    confusion = edit_distance(words_orig, words_pert) / max(|orig|, |pert|)

Higher confusion = Whisper heard meaningfully different words on the
perturbed file (i.e. the poisoning is working).

Whisper settings here are tuned for **speed and fairness**, not maximum
accuracy. The same settings are applied to both files, so the relative
comparison is meaningful even if absolute transcription quality is lower.
"""

import re
from typing import List

from .. import models
from ..config import WHISPER_MAX_SECONDS

_APOSTROPHES = re.compile(r"['\u2019]")
_PUNCTUATION = re.compile(r"[^\w\s]|_")


def transcribe(audio_path: str) -> str:
    """Transcribe audio to a single space-joined string.

    Uses deterministic greedy decoding (``beam_size=1``, ``temperature=0``:
    no sampling fallback, so the same audio always gives the same text and
    an identical file scores 0 confusion), VAD filtering (skips long
    instrumental sections), and only the first ``WHISPER_MAX_SECONDS``
    of the file — typically the first verse and chorus.
    """
    model = models.whisper()
    segments, _ = model.transcribe(
        audio_path,
        beam_size=1,
        temperature=0.0,
        vad_filter=True,
        clip_timestamps=[0, WHISPER_MAX_SECONDS],
    )
    return " ".join(seg.text.strip() for seg in segments)


def normalise(text: str) -> List[str]:
    """Lower-case, strip punctuation and split into word tokens.

    Apostrophes are dropped so ``Don't`` and ``dont`` match; every other
    punctuation mark becomes a word break. Unicode-aware.
    """
    text = _APOSTROPHES.sub("", text.casefold())
    return _PUNCTUATION.sub(" ", text).split()


def confusion(original_text: str, perturbed_text: str) -> float:
    """Normalised word edit distance in ``[0, 1]``.

    Args:
        original_text:  Transcription of the original audio.
        perturbed_text: Transcription of the perturbed audio.

    Returns:
        0.0 when the normalised transcriptions are identical, 1.0 when no
        word lines up. Word order and repeats count. Symmetric in its two
        arguments.
    """
    words_o = normalise(original_text)
    words_p = normalise(perturbed_text)

    longest = max(len(words_o), len(words_p))
    if longest == 0:
        return 0.0
    return _edit_distance(words_o, words_p) / longest


def _edit_distance(a: List[str], b: List[str]) -> int:
    """Levenshtein distance between two token lists (insert/delete/substitute)."""
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,                         # delete
                current[j - 1] + 1,                      # insert
                previous[j - 1] + (token_a != token_b),  # substitute
            ))
        previous = current
    return previous[-1]
