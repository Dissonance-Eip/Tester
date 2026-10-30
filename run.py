"""
Command-line entry point.

Usage:
    python3 run.py original.wav perturbed.wav [--output results.json]
                   [--seconds N] [--device {cpu,mps,cuda}]

Outputs a console summary and writes the JSON report to ``--output``
(``results.json`` in the current directory by default).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

DEVICES = ("cpu", "mps", "cuda")
DEFAULT_OUTPUT = "results.json"

# Known, harmless warnings from third-party code, matched by message so
# anything new still shows up.
_IGNORED_WARNINGS = (
    (r"pkg_resources is deprecated", UserWarning),            # sacred, via MAEST
    (r"`torch\.jit\.(script|interface)` is deprecated", FutureWarning),  # MAEST
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a perturbation confuses AI models while staying inaudible."
    )
    parser.add_argument("original", type=Path, help="clean reference audio file")
    parser.add_argument("perturbed", type=Path, help="perturbed audio file")
    parser.add_argument(
        "--output", type=Path, default=Path(DEFAULT_OUTPUT),
        help=f"JSON report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seconds", type=int, default=None,
        help="cap the audio analysed per model (lowers the 30 s MERT/MAEST and "
             "60 s Whisper windows, never raises them)",
    )
    parser.add_argument(
        "--device", choices=DEVICES, default=None,
        help="device for the MERT encoder (default: mps if available, else cpu); "
             "Whisper and MAEST always run on cpu",
    )
    return parser.parse_args(argv)


def _input_errors(args: argparse.Namespace) -> list[str]:
    """Problems with the arguments that can be found without loading torch."""
    errors = [
        f"{label} file not found: {path}"
        for label, path in (("original", args.original), ("perturbed", args.perturbed))
        if not path.is_file()
    ]
    if args.seconds is not None and args.seconds <= 0:
        errors.append(f"--seconds must be a positive integer, got {args.seconds}")
    if not args.output.parent.is_dir():
        errors.append(f"output directory does not exist: {args.output.parent}")
    return errors


def _device_error(device: str | None) -> str | None:
    """Return an error if the requested device is not available here."""
    if device in (None, "cpu"):
        return None
    import torch

    available = torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    return None if available else f"--device {device} is not available on this machine"


def _quiet_third_party_noise() -> None:
    for message, category in _IGNORED_WARNINGS:
        warnings.filterwarnings("ignore", message=message, category=category)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    errors = _input_errors(args) or [e for e in [_device_error(args.device)] if e]
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2

    # config binds these at import time, so set them before importing tester.
    if args.device:
        os.environ["TESTER_DEVICE"] = args.device
    if args.seconds:
        os.environ["TESTER_MAX_SECONDS"] = str(args.seconds)

    _quiet_third_party_noise()

    # Import lazily so bad arguments fail fast without loading torch and the models.
    from tester import run

    run(str(args.original), str(args.perturbed), str(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
