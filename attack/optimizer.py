"""
Projected gradient-descent attack loop.

Crafts a perturbation ``δ`` that pushes a song's MERT embedding away from its
clean embedding (maximising the Tester's ``genre_conf`` cosine distance) while
holding ``δ`` at a fixed energy budget so the SNR stays at the target.

The energy constraint is the whole trick: without it the optimiser would just
make ``δ`` loud (easy to fool a model when you're allowed to be audible). By
projecting ``δ`` onto a fixed-RMS sphere every step, we force it to find the
*quiet, clever* perturbation instead of the *loud, dumb* one.

SNR relationship (matches ``tester/metrics/snr.py``): with
``perturbed = song + δ``, the metric's noise is ``orig - pert = -δ``, so

    SNR_dB = 20·log10( RMS(song) / RMS(δ) )

Therefore fixing ``RMS(δ) = RMS(song) / 10^(SNR/20)`` pins the SNR exactly.
"""

import math

import torch

from tester.config import DEVICE


_N_FFT = 2048
_HOP = 512


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt()


def _spectral_envelope(song: torch.Tensor) -> torch.Tensor:
    """Per-bin/per-frame magnitude spectrum of the music — the masking ceiling.

    Noise is only inaudible where the music is loud enough to cover it. The
    song's own STFT magnitude is a cheap, effective proxy for that masking
    threshold: large where the music has energy, ~0 in the quiet/exposed bands.
    """
    win = torch.hann_window(_N_FFT, device=song.device)
    S = torch.stft(song, n_fft=_N_FFT, hop_length=_HOP, window=win, return_complex=True)
    return S.abs()


def _perceptual_project(delta: torch.Tensor, env: torch.Tensor, length: int,
                        mask_scale: float) -> torch.Tensor:
    """Clamp δ's spectrum under ``env * mask_scale``, preserving its phase.

    This pushes noise *under* the music's spectral envelope: where the song is
    quiet, the ceiling is ~0 so the noise is removed there (the bands where you'd
    otherwise hear it). Where the song is loud, noise is allowed (it's masked).
    """
    win = torch.hann_window(_N_FFT, device=delta.device)
    D = torch.stft(delta, n_fft=_N_FFT, hop_length=_HOP, window=win, return_complex=True)
    mag = D.abs()
    phase = torch.angle(D)
    ceiling = env * mask_scale
    mag = torch.minimum(mag, ceiling)
    D = torch.polar(mag, phase)
    return torch.istft(D, n_fft=_N_FFT, hop_length=_HOP, window=win, length=length)


def attack(song: torch.Tensor, native_sr: int, target, *, iters: int = 300,
           lr: float = 1e-3, target_snr_db: float = 30.0, log_every: int = 25,
           perceptual: bool = False, mask_scale: float = 0.15):
    """Run the PGD attack and return the perturbation ``δ``.

    Args:
        song:          1-D mono waveform in [-1, 1].
        native_sr:     Sample rate of ``song``.
        target:        A target exposing ``embed`` / ``clean_embed`` (MertTarget).
        iters:         Number of gradient steps.
        lr:            Adam learning rate for ``δ``.
        target_snr_db: SNR the perturbation is pinned to (higher = quieter).
        log_every:     Print cadence.
        perceptual:    If True, shape δ under the music's spectral envelope so it
                       hides under the audio (inaudible) instead of flat noise.
        mask_scale:    In perceptual mode, how far below the music's spectrum to
                       keep δ (smaller = quieter / more hidden).

    Returns:
        ``δ`` as a 1-D CPU tensor the same length as ``song``.
    """
    song = song.to(DEVICE)
    clean = target.clean_embed(song, native_sr)
    length = song.shape[-1]

    # RMS budget that pins SNR to the target (see module docstring).
    budget = (_rms(song) / (10.0 ** (target_snr_db / 20.0))).item()
    env = _spectral_envelope(song) if perceptual else None

    delta = torch.zeros_like(song, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    mode = f"perceptual (mask_scale={mask_scale})" if perceptual else "flat-RMS"
    print(f"  song RMS={_rms(song).item():.5f}  δ budget RMS={budget:.6f} "
          f"(SNR target {target_snr_db:.0f} dB)  [{mode}]\n")

    for i in range(iters):
        opt.zero_grad()
        emb = target.embed(song + delta, native_sr)
        cos = torch.nn.functional.cosine_similarity(
            emb.unsqueeze(0), clean.unsqueeze(0)
        ).mean()
        cos.backward()           # minimise cosine similarity → maximise distance
        opt.step()

        with torch.no_grad():
            if perceptual:
                # Hide δ under the music's spectrum, then (if still too loud)
                # scale DOWN to the SNR budget — never up, so it stays masked.
                projected = _perceptual_project(delta, env, length, mask_scale)
                r = _rms(projected).item()
                if r > budget and r > 1e-12:
                    projected.mul_(budget / r)
                delta.copy_(projected)
            else:
                # Flat: project onto the fixed-RMS sphere (full SNR budget).
                r = _rms(delta).item()
                if r > 1e-12:
                    delta.mul_(budget / r)

        if i % log_every == 0 or i == iters - 1:
            with torch.no_grad():
                conf = 1.0 - cos.item()                     # the genre_conf metric
                snr = 20.0 * math.log10(_rms(song).item() / max(_rms(delta).item(), 1e-12))
            print(f"  iter {i:4d}  genre_conf={conf:6.3f}  SNR={snr:5.1f} dB")

    return delta.detach().cpu()


def attack_universal(segments, native_sr, target, *, iters: int = 200, lr: float = 1e-3,
                     target_snr_db: float = 30.0, log_every: int = 10):
    """Train ONE shared δ that fools the model across MANY songs (a UAP).

    Every training segment gets the *same* δ added to it. Each iteration sums
    the attack gradient from all segments onto the single shared δ, so δ learns
    to be adversarial on average — not memorised to one song. Generalisation is
    then judged by applying δ to a held-out song (see run_uap.py).

    Args:
        segments:      List of equal-length 1-D mono waveforms (training set).
        native_sr:     Sample rate shared by all segments.
        target:        MertTarget.
        iters/lr/target_snr_db/log_every: as in ``attack``.

    Returns:
        ``δ`` (1-D CPU tensor, length = segment length) — apply to any song.
    """
    segs = [s.to(DEVICE) for s in segments]
    cleans = [target.clean_embed(s, native_sr) for s in segs]
    length = segs[0].shape[-1]

    # Budget from the average training-song loudness → ~target SNR on average.
    avg_rms = sum(_rms(s).item() for s in segs) / len(segs)
    budget = avg_rms / (10.0 ** (target_snr_db / 20.0))

    delta = torch.zeros(length, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    print(f"  {len(segs)} training songs  δ budget RMS={budget:.6f} "
          f"(avg SNR target {target_snr_db:.0f} dB)\n")

    for i in range(iters):
        opt.zero_grad()
        total_conf = 0.0
        # Gradient accumulation: one song at a time keeps peak memory to a
        # single backward while still summing every song's gradient onto δ.
        for seg, clean in zip(segs, cleans):
            emb = target.embed(seg + delta, native_sr)
            cos = torch.nn.functional.cosine_similarity(
                emb.unsqueeze(0), clean.unsqueeze(0)
            ).mean()
            cos.backward()
            total_conf += 1.0 - cos.item()
        opt.step()

        with torch.no_grad():
            r = _rms(delta).item()
            if r > 1e-12:
                delta.mul_(budget / r)

        if i % log_every == 0 or i == iters - 1:
            print(f"  iter {i:4d}  avg genre_conf={total_conf / len(segs):6.3f}")

    return delta.detach().cpu()
