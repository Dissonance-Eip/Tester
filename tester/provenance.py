"""
Environment provenance for ``results.json["run"]``.

Records the package versions and model revisions a score was produced
with, so two results can be compared knowing whether the stack matched.
"""

import json
import platform
from importlib import metadata

from .config import MERT_REVISION, WHISPER_REVISION

# Packages whose version can change a score.
TRACKED_PACKAGES = (
    "torch",
    "torchaudio",
    "transformers",
    "huggingface_hub",
    "faster-whisper",
    "ctranslate2",
    "maest",
    "numpy",
    "librosa",
    "soundfile",
)

NOT_INSTALLED = "not installed"


def package_versions() -> dict[str, str]:
    """Installed version of each tracked package, plus the Python version."""
    versions = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = NOT_INSTALLED
    return versions


def model_revisions() -> dict[str, str | None]:
    """Pinned Hub revisions, plus the installed MAEST commit."""
    return {
        "lyrics_transcriber": WHISPER_REVISION,
        "music_encoder":      MERT_REVISION,
        "genre_classifier":   maest_commit(),
    }


def maest_commit() -> str | None:
    """Git commit MAEST was installed from, read from its PEP 610 record.

    Read from the install rather than repeated in ``config`` so it can't
    drift from requirements.in. ``None`` if MAEST wasn't installed from git.
    """
    try:
        raw = metadata.distribution("maest").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw).get("vcs_info", {}).get("commit_id")
    except (json.JSONDecodeError, AttributeError):
        return None
