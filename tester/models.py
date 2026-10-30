"""
Lazy-loaded model singletons.

Each model is loaded the first time it's requested and cached. Importing
this module does *not* load anything — keep startup fast.

The three models we use:

* **Whisper** (faster-whisper, CPU only) — lyrics transcription
* **MERT**   (HuggingFace, GPU when available) — music latent encoder
* **MAEST**  (palonso/maest, CPU only) — Discogs genre classifier

Why some are CPU-only:

* faster-whisper's CTranslate2 backend doesn't support MPS.
* MAEST's internal melspectrogram module lazy-initialises its STFT window
  on CPU and doesn't track device, so we keep the whole model on CPU.
"""

from functools import cache
from typing import Any

from faster_whisper import WhisperModel
from maest import get_maest
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from .config import (
    DEVICE,
    ENCODER_MODEL,
    MAEST_ARCH,
    MERT_REVISION,
    WHISPER_REVISION,
    WHISPER_SIZE,
)

# Each loader runs once; ``functools.cache`` keeps the model for later calls.


@cache
def whisper() -> WhisperModel:
    """Return the Whisper transcription model, loading it on first call."""
    print(f"  Loading Whisper {WHISPER_SIZE} (cpu)...")
    return WhisperModel(
        WHISPER_SIZE, device="cpu", compute_type="int8", revision=WHISPER_REVISION
    )


@cache
def mert() -> tuple[Any, Any]:
    """Return ``(model, feature_extractor)`` for the MERT music encoder."""
    print(f"  Loading MERT music encoder ({DEVICE})...")
    model = AutoModel.from_pretrained(
        ENCODER_MODEL, revision=MERT_REVISION, trust_remote_code=True
    )
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        ENCODER_MODEL, revision=MERT_REVISION, trust_remote_code=True
    )
    return model.to(DEVICE).eval(), extractor


@cache
def maest() -> Any:
    """Return the MAEST genre classifier."""
    print("  Loading MAEST genre classifier (cpu)...")
    return get_maest(arch=MAEST_ARCH).eval()


def load_all() -> None:
    """Eagerly load every model. Useful so the user sees all download
    progress up-front rather than scattered between pipeline stages."""
    whisper()
    mert()
    maest()
    print("  Models ready.\n")
