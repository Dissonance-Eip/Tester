# Music Perturbation Tester

A local evaluation harness for audio poisoning perturbations.

A successful poison is **inaudible to humans** but **confuses AI models**.
This tester quantifies both sides of that trade-off so you can iterate
on the perturbation algorithm with real numbers.

---

## What it measures

| Metric | Tool | A good poison... |
|--------|------|------------------|
| **Human audibility (SNR)** | RMS comparison | ≥ 30 dB — listener can't hear the perturbation |
| **Lyrics confusion** | Whisper `large-v3-turbo` | ≥ 50% — Whisper transcribes the wrong words |
| **Genre label** | MAEST (Discogs 400-tag) | label changes — AI calls it a different genre |
| **Encoder confusion** | MERT v1 330M | ≥ 30% cosine distance — music-understanding AI sees a different track |

All three confusion metrics must move while SNR stays high. If SNR drops,
humans hear artifacts; if the AI scores don't move, the noise isn't
actually adversarial.

---

## Requirements

- macOS on Apple Silicon (M1 / M2 / M3 / M4)
- 16 GB RAM minimum (32 GB recommended)
- ~10 GB free disk for model weights

That's it — no prerequisites. `setup.sh` installs Homebrew, pyenv,
Python 3.11.9, ffmpeg, and everything else.

---

## Setup

```bash
chmod +x setup.sh
./setup.sh
```

The script handles, in order:

1. Xcode Command Line Tools
2. Homebrew (if missing)
3. `ffmpeg`, `pyenv`, and Python build deps
4. Python 3.11.9 via pyenv
5. A project-local virtualenv at `./venv/`
6. PyTorch + faster-whisper + transformers + MAEST
7. `PYTORCH_ENABLE_MPS_FALLBACK=1` in your shell rc
8. A dependency sanity check (`test.py`)

When it finishes you'll see:
```
============================================
  All done! Ready to test perturbations.
============================================
```

---

## Usage

```bash
source venv/bin/activate
python3 run.py original.wav poisoned.wav
```

Outputs:
- A terminal summary box with the metric values
- `results.json` with the full structured report

Sanity-check the pipeline by passing the same file twice — SNR should be
`inf dB`, confusions near 0%, genre label unchanged:

```bash
python3 run.py sounds/sound.wav sounds/sound.wav
```

---

## First-run downloads

These are cached in `~/.cache/huggingface/` after the first run:

| Model | Size |
|-------|------|
| Whisper `large-v3-turbo` | ~1.6 GB |
| MERT `v1-330M`           | ~1.3 GB |
| MAEST `discogs-129e`     | ~340 MB |

Total ~3.3 GB. Set `HF_TOKEN` in your shell for faster downloads.

---

## Project layout

```
music-perturbation-tester/
├── run.py                 # CLI entry point
├── test.py                # dependency sanity check
├── setup.sh               # full environment installer
├── cleanup.sh             # remove venv + model caches
├── requirements.txt
├── README.md              # this file
└── tester/                # package
    ├── __init__.py
    ├── config.py          # constants & thresholds
    ├── audio.py           # load / mono / resample / trim
    ├── models.py          # lazy model singletons
    ├── runner.py          # orchestrates the pipeline
    ├── report.py          # console + JSON output
    └── metrics/
        ├── snr.py         # human-perception metric
        ├── lyrics.py      # Whisper-based lyrics confusion
        └── genre.py       # MAEST classifier + MERT encoder
```

Change a threshold or swap a model in `tester/config.py`. Add a new
metric by dropping a file in `tester/metrics/` and calling it from
`runner.py`.

---

## Cleanup

To remove the venv, downloaded models, or the pyenv Python install:

```bash
./cleanup.sh
```

Homebrew packages are left alone — they're used system-wide.

---

## Models used

| Model | Job | Where |
|-------|-----|-------|
| `large-v3-turbo` (Whisper) | Lyrics transcription | CPU (CTranslate2 — no MPS) |
| `m-a-p/MERT-v1-330M` | Music understanding encoder | **MPS** (GPU) |
| `mtg-upf/discogs-maest-30s-pw-129e` | Discogs genre classification | CPU (preprocessor bug on MPS) |

---

## Interpreting results

If the perturbation isn't working you'll see one of these patterns:

| Symptom | Likely cause |
|---------|--------------|
| SNR < 20 dB | Perturbation amplitude too large |
| All confusions ~0% | Noise isn't adversarial — random/uniform doesn't target AI features |
| Lyrics confused, encoder unchanged | Hurting ASR but not music models |
| Encoder confused, lyrics unchanged | Hurting music models but transcription still works |

A working adversarial perturbation (Nightshade/Glaze-style) does
gradient-based optimization against a target model's embedding while
constraining L∞ — that's what'll move all three numbers at once.
