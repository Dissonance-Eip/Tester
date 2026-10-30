"""
Configuration constants.

All tunable parameters live here so the rest of the codebase has no magic
numbers. Change a threshold or swap a model by editing one line.
"""

import torch

# ── Compute device ───────────────────────────────────────────
# Apple Silicon Metal Performance Shaders if available, else CPU.
DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Model identifiers ────────────────────────────────────────
WHISPER_SIZE:  str = "large-v3-turbo"           # faster-whisper model size
ENCODER_MODEL: str = "m-a-p/MERT-v1-330M"       # MERT music understanding encoder
MAEST_ARCH:    str = "discogs-maest-30s-pw-129e"  # MAEST Discogs genre classifier

# ── Sample rates expected by each model (Hz) ─────────────────
MERT_SR:  int = 24_000
MAEST_SR: int = 16_000

# ── Audio length caps for inference (seconds) ────────────────
MAX_AUDIO_SECONDS:    int = 30   # MERT + MAEST window
WHISPER_MAX_SECONDS:  int = 60   # Whisper transcription window

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

# ── Adversarial attack (attack/) ─────────────────────────────
# STFT of attack/optimizer.py's perceptual projection. It only approximates
# core's MaskingStage (core/include/audio/MaskingStage.hpp): the frame is the
# same 2048-point Hann FFT, but the hop is 512 (core: 1024), the ceiling is
# the song's own magnitude × ATTACK_MASK_SCALE (core: Bark-band
# psychoacoustic thresholds), and it clamps |δ| per bin (core: the per-bin
# magnitude difference between perturbed and clean).
ATTACK_STFT_FRAME: int   = 2048   # same frame size as core's MaskingStage
ATTACK_STFT_HOP:   int   = 512    # core uses 1024 (50% overlap)
ATTACK_MASK_SCALE: float = 0.15   # ceiling = song magnitude × this
