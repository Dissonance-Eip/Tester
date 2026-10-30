"""Lyrics confusion: normalisation and order-aware comparison (issue #2)."""

import pytest

from tester.metrics import lyrics
from tester.metrics.lyrics import confusion


def test_punctuation_and_case_are_ignored():
    assert confusion("Hello, world! Don't stop.", "hello world dont stop") == 0.0


def test_reordered_words_count_as_different():
    # Same bag of words, different order: a set comparison scores this 0.
    assert confusion("one two three four", "four three two one") == pytest.approx(1.0)


def test_repeated_words_count():
    # Dropping two of three repeats is a real transcription change.
    assert confusion("la la la love", "la love") == pytest.approx(0.5)


def test_single_substitution_is_one_edit():
    assert confusion("I walk the line", "I talk the line") == pytest.approx(0.25)


def test_is_symmetric_and_bounded():
    a, b = "the quick brown fox", "a quick fox jumps over"
    assert confusion(a, b) == confusion(b, a)
    assert 0.0 <= confusion(a, b) <= 1.0


@pytest.mark.parametrize(
    ("original", "perturbed", "expected"),
    [("", "", 0.0), ("", "words here", 1.0), ("words here", "", 1.0), ("...", "!!", 0.0)],
)
def test_empty_or_punctuation_only_transcripts(original, perturbed, expected):
    assert confusion(original, perturbed) == expected


def test_transcription_is_deterministic(monkeypatch):
    # Whisper's default temperature fallback samples, so the same file could
    # transcribe differently twice and score as "confused" against itself.
    calls = []

    class FakeWhisper:
        def transcribe(self, audio_path, **kwargs):
            calls.append(kwargs)
            return iter(()), None

    monkeypatch.setattr(lyrics.models, "whisper", FakeWhisper)

    lyrics.transcribe("song.wav")

    assert calls[0].get("temperature") == 0.0
