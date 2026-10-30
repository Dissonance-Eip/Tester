"""
Dependency sanity check.

Run after ``setup.sh`` to verify the environment is healthy:

    python3 test.py
"""

import subprocess
import sys
from collections.abc import Callable


def _check(label: str, fn: Callable[[], object]) -> bool:
    """Run ``fn``; print the label if it succeeds, or the error if it doesn't."""
    try:
        fn()
        print(f"  {label}")
        return True
    except (ImportError, OSError, RuntimeError) as exc:  # missing module, dylib or binary
        print(f"  {label} — {exc}")
        return False


def _check_torch() -> None:
    import torch
    print(f"       MPS available: {torch.backends.mps.is_available()}")


def _check_ffmpeg() -> None:
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg not found")


def main() -> int:
    print("\n── Checking dependencies ──────────────────────")

    checks: list[tuple[str, Callable[[], object]]] = [
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
        print(f"  All {total}/{total} checks completed.\n")
        return 0

    print(f"  {passed}/{total} checks completed — see items above then re-run.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
