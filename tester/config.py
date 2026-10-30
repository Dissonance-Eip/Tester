"""
Configuration constants.

All tunable parameters live here so the rest of the codebase has no magic
numbers. Change a threshold or swap a model by editing one line.
"""

import os

import torch

# ── Command-line overrides ───────────────────────────────────
# run.py sets these before importing the package, because the values below
# are bound at import time by the modules that use them.
DEVICE_ENV:      str = "TESTER_DEVICE"
MAX_SECONDS_ENV: str = "TESTER_MAX_SECONDS"


def _seconds_cap(default: int) -> int:
    """``default``, lowered (never raised) by ``$TESTER_MAX_SECONDS`` when set."""
    cap = os.environ.get(MAX_SECONDS_ENV)
    return min(default, int(cap)) if cap else default


# ── Compute device ───────────────────────────────────────────
# Apple Silicon Metal Performance Shaders if available, else CPU.
DEVICE: str = os.environ.get(DEVICE_ENV) or ("mps" if torch.backends.mps.is_available() else "cpu")

# ── Model identifiers ────────────────────────────────────────
WHISPER_SIZE:  str = "large-v3-turbo"           # faster-whisper model size
ENCODER_MODEL: str = "m-a-p/MERT-v1-330M"       # MERT music understanding encoder
MAEST_ARCH:    str = "discogs-maest-30s-pw-129e"  # MAEST Discogs genre classifier

# ── Pinned Hugging Face revisions (commit SHAs) ──────────────
# Loading by SHA means a Hub update can't silently change the weights, and
# for MERT (trust_remote_code=True) it also pins the model code that runs
# locally. Bump a SHA deliberately and note it: scores move with it.
# MAEST weights are not on the Hub: they come from a GitHub release URL
# fixed by the MAEST commit pinned in requirements.in.
# Whisper: mobiuslabsgmbh/faster-whisper-large-v3-turbo; MERT: m-a-p/MERT-v1-330M.
WHISPER_REVISION: str = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
MERT_REVISION:    str = "5240c2708a5acaee1007f43fb9735c7dcd0b78c9"

# ── Whisper decoding ─────────────────────────────────────────
# Fixed settings so the same audio always gives the same text.
WHISPER_LANGUAGE:    str | None = "en"      # None = auto-detect per file
WHISPER_TEMPERATURE: float = 0.0           # greedy, no sampling fallback
WHISPER_CONDITION_ON_PREVIOUS_TEXT: bool = False  # no drift from earlier segments

# ── Sample rates expected by each model (Hz) ─────────────────
MERT_SR:  int = 24_000
MAEST_SR: int = 16_000

# ── Audio length caps for inference (seconds) ────────────────
MAX_AUDIO_SECONDS:    int = _seconds_cap(30)   # MERT + MAEST window
WHISPER_MAX_SECONDS:  int = _seconds_cap(60)   # Whisper transcription window

# ── SNR measurement ──────────────────────────────────────────
# Measure the audio every model sees (the longest model window), not the
# whole file, so a perturbation confined to that window is not diluted.
SNR_WINDOW_SECONDS:  int   = max(MAX_AUDIO_SECONDS, WHISPER_MAX_SECONDS)
SNR_SEGMENT_SECONDS: float = 0.5    # frame length for segmental SNR
SILENCE_RMS:         float = 1e-4   # ≈ -80 dBFS; quieter originals count as silent

# ── Pass / fail thresholds ───────────────────────────────────
# SNR is in decibels; higher = more inaudible to humans.
SNR_GOOD: float = 30.0   # imperceptible to a human listener
SNR_WARN: float = 20.0   # borderline; below this is clearly audible

# Confusion scores are in [0, 1]; higher = AI is more confused.
LYRICS_CONFUSION_GOOD: float = 0.5   # ≥50% word divergence = success
GENRE_CONFUSION_GOOD:  float = 0.3   # cosine distance ≥ 0.3 = success
