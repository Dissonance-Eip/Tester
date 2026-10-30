"""
Universal Adversarial Perturbation (UAP) trainer + held-out test.

Trains ONE shared δ across several songs, then applies it to a song it never
saw, so you can measure whether it *generalises* (the whole point of Path B).

Usage:
    python3 -m attack.run_uap --train sounds/a.wav sounds/b.wav sounds/c.wav \
                              --test sounds/d.wav

Writes into sounds/:
    uap_delta.wav              the universal perturbation (the shippable artifact)
    <test>_uap_clean.wav       held-out song, mono reference (first N s)
    <test>_uap_poisoned.wav    held-out song + δ  → feed to the Tester

Validate generalisation (the make-or-break number):
    python3 run.py sounds/<test>_uap_clean.wav sounds/<test>_uap_poisoned.wav

If the held-out song's encoder confusion stays high, the UAP generalises.
"""

import argparse
import os
import sys
import warnings

import soundfile as sf
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

from tester.audio import load_mono  # noqa: E402
from tester.config import MAX_AUDIO_SECONDS, SNR_GOOD  # noqa: E402
from attack.optimizer import attack_universal  # noqa: E402
from attack.targets import MertTarget  # noqa: E402


def load_songs(train, test, seconds):
    """Load each song once, as mono, keeping its first ``seconds``.

    Every song must share one sample rate: the shared δ is added sample by
    sample, so mixing rates would silently misalign it. Training segments
    are truncated to the shortest one.

    Returns:
        ``(train_segments, test_segment, sample_rate)``.

    Raises:
        ValueError: if the songs do not all have the same sample rate.
    """
    segs, sr = [], None
    for path in [*train, test]:
        seg, rate = load_mono(path, seconds=seconds)
        if sr is not None and rate != sr:
            raise ValueError(f"{path} has sample rate {rate} Hz, expected {sr} Hz "
                             "like the first training song; resample it first")
        sr = rate
        segs.append(seg)
    *train_segs, test_seg = segs
    n = min(s.shape[-1] for s in train_segs)
    return [s[:n] for s in train_segs], test_seg, sr


def main() -> int:
    ap = argparse.ArgumentParser(description="Universal adversarial perturbation (UAP) trainer")
    ap.add_argument("--train", nargs="+", required=True, help="Training song WAVs (2+)")
    ap.add_argument("--test", required=True, help="Held-out song WAV (not in --train)")
    ap.add_argument("--seconds", type=int, default=MAX_AUDIO_SECONDS, help="Window length per song")
    ap.add_argument("--iters", type=int, default=200, help="Gradient steps")
    ap.add_argument("--snr", type=float, default=SNR_GOOD, help="Avg loudness budget in dB")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    args = ap.parse_args()

    print(f"\nLoading {len(args.train)} training songs ...")
    try:
        segs, test_seg, sr = load_songs(args.train, args.test, args.seconds)
    except ValueError as exc:
        ap.error(str(exc))
    for p, seg in zip(args.train, segs):
        print(f"  {os.path.basename(p)}  {seg.shape[-1] / sr:.1f}s")

    print("\nLoading MERT ...")
    target = MertTarget()

    print(f"\nTraining universal δ ({args.iters} iters) ...")
    delta = attack_universal(segs, sr, target, iters=args.iters, lr=args.lr,
                             target_snr_db=args.snr)

    # Held-out validation — apply δ to a song it never saw.
    m = min(test_seg.shape[-1], delta.shape[-1])
    clean = test_seg[:m]
    poisoned = torch.clamp(clean + delta[:m], -1.0, 1.0)

    sounds = os.path.join(_ROOT, "sounds")
    os.makedirs(sounds, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.test))[0]
    delta_path = os.path.join(sounds, "uap_delta.wav")
    clean_path = os.path.join(sounds, f"{name}_uap_clean.wav")
    poison_path = os.path.join(sounds, f"{name}_uap_poisoned.wav")

    sf.write(delta_path, delta.numpy(), sr, subtype="FLOAT")
    sf.write(clean_path, clean.numpy(), sr, subtype="FLOAT")
    sf.write(poison_path, poisoned.numpy(), sr, subtype="FLOAT")

    print("\nWrote → sounds/")
    print(f"  uap_delta.wav            (the universal perturbation)")
    print(f"  {name}_uap_clean.wav")
    print(f"  {name}_uap_poisoned.wav")
    print("\nHELD-OUT generalisation test (the number that matters):")
    print(f"  python3 run.py {os.path.relpath(clean_path)} {os.path.relpath(poison_path)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
