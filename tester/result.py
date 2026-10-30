"""
Result schema for ``results.json``.

``RunResult`` holds the raw measurements of one run. ``to_dict`` turns
them into the JSON report, one builder per section. Everything here is
pure apart from the ``run`` section, which reads the clock and the
installed package versions.
"""

import datetime
import math
import os
from dataclasses import dataclass
from typing import Any

from . import provenance
from .config import (
    DEVICE,
    ENCODER_MODEL,
    GENRE_CONFUSION_GOOD,
    LYRICS_CONFUSION_GOOD,
    MAEST_ARCH,
    SNR_GOOD,
    SNR_WARN,
    SNR_WINDOW_SECONDS,
    WHISPER_SIZE,
)
from .metrics.snr import SnrResult

Genres = list[tuple[str, float]]

ABOUT = (
    "Audio poisoning effectiveness report. A successful perturbation is "
    "inaudible to humans (high SNR) and confuses AI on both lyrics and genre."
)

_SNR_INTERPRETATION = {
    "imperceptible": "Imperceptible to a human listener.",
    "borderline":    "Borderline — some listeners may notice.",
    "noticeable":    "Clearly audible artifacts — humans hear the perturbation.",
}
_SILENT_INTERPRETATION = "Original is silent; any perturbation over silence is audible."


def snr_label(snr_db: float) -> str:
    """Human-readable band for an SNR value: the only place the ladder lives."""
    if snr_db >= SNR_GOOD:
        return "imperceptible"
    if snr_db >= SNR_WARN:
        return "borderline"
    return "noticeable"


@dataclass(frozen=True)
class RunResult:
    """Raw measurements from one evaluation run."""

    original_path: str
    perturbed_path: str
    snr: SnrResult
    original_lyrics: str
    perturbed_lyrics: str
    lyrics_confusion: float
    original_genres: Genres
    perturbed_genres: Genres
    encoder_confusion: float

    # ── Gates ────────────────────────────────────────────────
    @property
    def snr_pass(self) -> bool:
        # The quietest frame decides audibility.
        return self.snr.segmental_min_db >= SNR_GOOD

    @property
    def lyrics_pass(self) -> bool:
        return self.lyrics_confusion >= LYRICS_CONFUSION_GOOD

    @property
    def encoder_pass(self) -> bool:
        return self.encoder_confusion >= GENRE_CONFUSION_GOOD

    @property
    def label_changed(self) -> bool:
        return _top(self.original_genres) != _top(self.perturbed_genres)

    @property
    def checks(self) -> dict[str, bool]:
        """Every success criterion listed in the README counts towards the verdict."""
        return {
            "snr":         self.snr_pass,
            "lyrics":      self.lyrics_pass,
            "encoder":     self.encoder_pass,
            "genre_label": self.label_changed,
        }

    # ── JSON report ──────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """The full ``results.json`` document."""
        return {
            "_about":           ABOUT,
            "run":              self.run_section(),
            "verdict":          self.verdict_section(),
            "human_perception": self.human_perception_section(),
            "ai_lyrics":        self.lyrics_section(),
            "ai_genre":         self.genre_section(),
        }

    def run_section(self) -> dict[str, Any]:
        return {
            "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
            "original_file":  os.path.abspath(self.original_path),
            "perturbed_file": os.path.abspath(self.perturbed_path),
            "device":         DEVICE,
            "models": {
                "lyrics_transcriber": WHISPER_SIZE,
                "genre_classifier":   MAEST_ARCH,
                "music_encoder":      ENCODER_MODEL,
            },
            "model_revisions": provenance.model_revisions(),
            "versions":        provenance.package_versions(),
        }

    def verdict_section(self) -> dict[str, Any]:
        checks = self.checks
        overall_pass = all(checks.values())
        return {
            "overall_pass":   overall_pass,
            "summary": (
                "Inaudible to humans, confusing to AI."
                if overall_pass
                else "See individual metrics below."
            ),
            "metrics_passed": sum(checks.values()),
            "metrics_total":  len(checks),
            "checks":         checks,
        }

    def human_perception_section(self) -> dict[str, Any]:
        label = snr_label(self.snr.segmental_min_db)
        silent_fail = self.snr.original_silent and not self.snr_pass
        return {
            "_description": "How audible the perturbation is to a human listener.",
            "window_seconds": SNR_WINDOW_SECONDS,
            "snr_db":       _db(self.snr.snr_db),
            "segmental_snr_db": {
                "min":  _db(self.snr.segmental_min_db),
                "mean": _db(self.snr.segmental_mean_db),
            },
            "gate":         "segmental_snr_db.min",
            "original_silent": self.snr.original_silent,
            "label":        label,
            "threshold":    {"good_above_db": SNR_GOOD, "warn_below_db": SNR_WARN},
            "pass":         self.snr_pass,
            "interpretation": (
                _SILENT_INTERPRETATION if silent_fail else _SNR_INTERPRETATION[label]
            ),
        }

    def lyrics_section(self) -> dict[str, Any]:
        return {
            "_description":     "Does Whisper hear different words on the perturbed file?",
            "confusion_score":  round(self.lyrics_confusion, 4),
            "threshold":        {"good_above": LYRICS_CONFUSION_GOOD},
            "pass":             self.lyrics_pass,
            "original_lyrics":  self.original_lyrics,
            "perturbed_lyrics": self.perturbed_lyrics,
            "interpretation": (
                "Whisper transcribes significantly different lyrics."
                if self.lyrics_pass
                else "Whisper transcribes the same lyrics."
            ),
        }

    def genre_section(self) -> dict[str, Any]:
        return {
            "_description": "Does AI genre classification and music encoding change?",
            "label_classifier": {
                "original_top_genres":  _genre_list(self.original_genres),
                "perturbed_top_genres": _genre_list(self.perturbed_genres),
                "original_top":   _top(self.original_genres),
                "perturbed_top":  _top(self.perturbed_genres),
                "label_changed":  self.label_changed,
                "pass":           self.label_changed,
            },
            "encoder_confusion": {
                "_description": "Cosine distance between MERT embeddings.",
                "score":        round(self.encoder_confusion, 4),
                "threshold":    {"good_above": GENRE_CONFUSION_GOOD},
                "pass":         self.encoder_pass,
            },
            "interpretation": (
                "AI music encoder sees a meaningfully different track."
                if self.encoder_pass
                else "AI music encoder sees essentially the same track."
            ),
        }


def _top(genres: Genres) -> str | None:
    """Top-ranked genre label, or ``None`` if the classifier returned none."""
    return genres[0][0] if genres else None


def _genre_list(genres: Genres) -> list[dict[str, Any]]:
    return [{"genre": g, "confidence": round(s, 4)} for g, s in genres]


def _db(value: float) -> float | str:
    """Round a dB value for JSON; infinities become ``"inf"`` / ``"-inf"``."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return round(value, 2)
