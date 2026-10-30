"""
Adversarial attack runner.

Crafts an (inaudible) adversarial perturbation against MERT and writes it next
to the input, in the sounds/ folder by default so you can open the files in
your editor:

    sounds/<name>_clean.wav     mono reference (what we attack)
    sounds/<name>_poisoned.wav  reference + δ  → feed this to the Tester
    sounds/<name>_delta.wav     δ alone, to hear the noise on its own

Common use (good defaults — inaudible, the Tester's MERT window, whole file):

    python3 -m attack.run_attack sounds/sound.wav

Then validate:

    python3 run.py sounds/sound_clean.wav sounds/sound_poisoned.wav

Knobs (all optional):
    --iters N     gradient steps per window      (default 100)
    --window S    attack window seconds          (default MAX_AUDIO_SECONDS, MERT eval)
    --seconds S   cap total length, 0 = whole    (default 0)
    --snr DB      loudness budget, higher=quieter(default SNR_GOOD)
    --mask F      perceptual hiding strength      (default ATTACK_MASK_SCALE, smaller=quieter)
    --flat        disable perceptual shaping (audible; for comparison only)
    --lr / --checkpoint   advanced
"""

import argparse
import os
import sys
import warnings

import soundfile as sf
import torch

# Silence PyTorch's benign STFT/iSTFT output-resize deprecation noise.
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

# Allow running both as `python3 -m attack.run_attack` and `python3 attack/run_attack.py`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tester.audio import load_mono  # noqa: E402
from tester.config import ATTACK_MASK_SCALE, MAX_AUDIO_SECONDS, SNR_GOOD  # noqa: E402
from attack.optimizer import attack  # noqa: E402
from attack.targets import MertTarget  # noqa: E402

# Windows shorter than this are left unperturbed (too short for MERT's frontend).
_MIN_WINDOW_SECONDS = 1


def _default_stem(song: str) -> str:
    """Output stem inside sounds/, named after the input file."""
    name = os.path.splitext(os.path.basename(song))[0]
    return os.path.join(_ROOT, "sounds", name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Adversarial perturbation generator (MERT)")
    ap.add_argument("song", help="Path to the input WAV")
    ap.add_argument("--iters", type=int, default=100, help="Gradient steps per window")
    ap.add_argument("--window", type=int, default=MAX_AUDIO_SECONDS,
                    help=f"Attack window in seconds (default {MAX_AUDIO_SECONDS} = MERT's eval window)")
    ap.add_argument("--seconds", type=int, default=0, help="Cap total length (0 = whole file)")
    ap.add_argument("--snr", type=float, default=SNR_GOOD,
                    help="Loudness budget in dB (higher=quieter)")
    ap.add_argument("--mask", type=float, default=ATTACK_MASK_SCALE,
                    help="Perceptual hiding strength (smaller = quieter / more hidden)")
    ap.add_argument("--flat", action="store_true",
                    help="Disable perceptual shaping (audible; comparison only)")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (advanced)")
    ap.add_argument("--checkpoint", action="store_true",
                    help="Gradient checkpointing — slower, only for very long windows")
    ap.add_argument("--out", default=None, help="Output stem (default: sounds/<name>)")
    args = ap.parse_args()

    stem = args.out or _default_stem(args.song)
    clean_path = f"{stem}_clean.wav"
    poison_path = f"{stem}_poisoned.wav"
    delta_path = f"{stem}_delta.wav"
    perceptual = not args.flat

    print(f"\nLoading {args.song} ...")
    mono, sr = load_mono(args.song, seconds=args.seconds if args.seconds > 0 else None)
    total = mono.shape[-1]
    print(f"  {total / sr:.1f}s mono @ {sr} Hz")

    print("\nLoading MERT ...")
    target = MertTarget(use_checkpointing=args.checkpoint)

    # Attack the file in windows so δ covers every second. Each window is
    # optimised against MERT independently; the pieces concatenate into one
    # full-length δ. The default window is the tester's MERT eval window
    # (MAX_AUDIO_SECONDS) so the attack transfers.
    win = args.window * sr
    n = (total + win - 1) // win
    print(f"\nAttacking {n} × {args.window}s window(s), {args.iters} iters, "
          f"{'perceptual' if perceptual else 'flat'} ...")

    parts = []
    for c in range(n):
        start, end = c * win, min((c + 1) * win, total)
        seg = mono[start:end]
        if seg.shape[-1] < _MIN_WINDOW_SECONDS * sr:
            parts.append(torch.zeros_like(seg))
            continue
        print(f"\n── window {c + 1}/{n}  ({start / sr:.0f}–{end / sr:.0f}s) ──")
        parts.append(attack(seg, sr, target, iters=args.iters, lr=args.lr,
                            target_snr_db=args.snr, perceptual=perceptual,
                            mask_scale=args.mask))

    delta = torch.cat(parts)[:total]
    poisoned = torch.clamp(mono + delta, -1.0, 1.0)

    os.makedirs(os.path.dirname(stem), exist_ok=True)
    sf.write(clean_path, mono.numpy(), sr, subtype="FLOAT")
    sf.write(poison_path, poisoned.numpy(), sr, subtype="FLOAT")
    sf.write(delta_path, delta.numpy(), sr, subtype="FLOAT")

    print(f"\nWrote → {os.path.relpath(os.path.dirname(stem), os.getcwd())}/")
    for p in (clean_path, poison_path, delta_path):
        print(f"  {os.path.basename(p)}")
    print(f"\nValidate:\n  python3 run.py {os.path.relpath(clean_path)} "
          f"{os.path.relpath(poison_path)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
