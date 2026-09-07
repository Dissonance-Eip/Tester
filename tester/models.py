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

from typing import Optional, Tuple

from faster_whisper import WhisperModel
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from maest import get_maest

from .config import WHISPER_SIZE, ENCODER_MODEL, MAEST_ARCH, DEVICE


# Module-level caches (None until first load)
_whisper:        Optional[WhisperModel] = None
_mert_model:     Optional[object]       = None
_mert_extractor: Optional[object]       = None
_maest_model:    Optional[object]       = None


def whisper() -> WhisperModel:
    """Return the Whisper transcription model, loading it on first call."""
    global _whisper
    if _whisper is None:
        print(f"  Loading Whisper {WHISPER_SIZE} (cpu)...")
        _whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _whisper


def mert() -> Tuple[object, object]:
    """Return ``(model, feature_extractor)`` for the MERT music encoder."""
    global _mert_model, _mert_extractor
    if _mert_model is None:
        print(f"  Loading MERT music encoder ({DEVICE})...")
        _mert_model = AutoModel.from_pretrained(ENCODER_MODEL, trust_remote_code=True)
        _mert_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            ENCODER_MODEL, trust_remote_code=True
        )
        _mert_model = _mert_model.to(DEVICE).eval()
    return _mert_model, _mert_extractor


def maest():
    """Return the MAEST genre classifier."""
    global _maest_model
    if _maest_model is None:
        print("  Loading MAEST genre classifier (cpu)...")
        _maest_model = get_maest(arch=MAEST_ARCH).eval()
    return _maest_model


def load_all() -> None:
    """Eagerly load every model. Useful so the user sees all download
    progress up-front rather than scattered between pipeline stages."""
    whisper()
    mert()
    maest()
    print("  Models ready.\n")
