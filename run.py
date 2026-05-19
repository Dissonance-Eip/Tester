"""
Command-line entry point.

Usage:
    python3 run.py original.wav perturbed.wav

Outputs a console summary and writes ``results.json`` in the current directory.
"""

import sys
import warnings
import logging


def _silence_third_party_noise() -> None:
    """Quiet harmless warnings from torch / transformers / audiocraft."""
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 run.py original.wav perturbed.wav")
        return 1

    _silence_third_party_noise()

    # Import lazily so `python3 run.py` with bad args doesn't drag in torch.
    from tester import run

    run(sys.argv[1], sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
