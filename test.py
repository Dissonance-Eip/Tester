"""
Dependency sanity check.

Run after ``setup.sh`` to verify the environment is healthy:

    python3 test.py
"""

import sys
import subprocess
from typing import Callable, List, Tuple


def _check(label: str, fn: Callable[[], None]) -> bool:
    """Run ``fn``; print ✅ if it succeeds, ❌ with the error if it doesn't."""
    try:
        fn()
        print(f"  ✅ {label}")
        return True
    except Exception as exc:
        print(f"  ❌ {label} — {exc}")
        return False


def _check_torch() -> None:
    import torch
    print(f"       MPS available: {torch.backends.mps.is_available()}")


def _check_ffmpeg() -> None:
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg not found")


def main() -> int:
    print("\n── Checking dependencies ──────────────────────")

    checks: List[Tuple[str, Callable[[], None]]] = [
        ("PyTorch",         _check_torch),
        ("torchaudio",      lambda: __import__("torchaudio")),
        ("soundfile",       lambda: __import__("soundfile")),
        ("numpy",           lambda: __import__("numpy")),
        ("faster-whisper",  lambda: __import__("faster_whisper")),
        ("transformers",    lambda: __import__("transformers")),
        ("maest",           lambda: __import__("maest")),
        ("ffmpeg",          _check_ffmpeg),
    ]

    results = [_check(label, fn) for label, fn in checks]
    passed  = sum(results)
    total   = len(results)

    print("───────────────────────────────────────────────")
    if passed == total:
        print(f"  All {total}/{total} checks passed — ready to go! 🎉\n")
        return 0

    print(f"  {passed}/{total} checks passed — fix the ❌ items above then re-run.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
