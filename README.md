# Music Perturbation Tester

A local evaluation harness for audio poisoning perturbations.

A successful poison is **inaudible to humans** but **confuses AI models**.
This tester quantifies both sides of that trade-off so you can iterate
on the perturbation algorithm with real numbers.

---

## What it measures

| Metric | Tool | A good poison... |
|--------|------|------------------|
| **Human audibility (SNR)** | RMS comparison over the first 60 s (what the models see), per 0.5 s frame | minimum frame SNR ≥ 30 dB — listener can't hear the perturbation |
| **Lyrics confusion** | Whisper `large-v3-turbo`, word edit distance (case and punctuation ignored) | ≥ 50% — Whisper transcribes the wrong words |
| **Genre label** | MAEST (Discogs 400-tag) | label changes — AI calls it a different genre |
| **Encoder confusion** | MERT v1 330M | ≥ 30% cosine distance — music-understanding AI sees a different track |

The verdict passes only when all four rows pass: the lyrics, genre label
and encoder must all move while SNR stays high. `results.json` reports
`metrics_passed` out of `metrics_total` (4), and the console prints the
verdict. If SNR drops, humans hear artifacts; if the AI scores don't move,
the noise isn't actually adversarial.

SNR is reported for the whole window and per frame (segmental min and
mean); the gate uses the minimum, so a short loud burst can't hide behind
a quiet average. If the original is silent, SNR is undefined: any
perturbation over it reports `-inf` and fails.

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

Options:

| Flag | Default | Effect |
|---|---|---|
| `--output PATH` | `results.json` | where the JSON report is written |
| `--seconds N` | model windows | caps the audio analysed (lowers the 30 s MERT/MAEST and 60 s Whisper windows, never raises them) |
| `--device {cpu,mps,cuda}` | `mps` if available, else `cpu` | device for MERT; Whisper and MAEST always run on CPU |

Missing input files, a non-positive `--seconds`, a missing output directory
or an unavailable device are reported before any model loads.

Sanity-check the pipeline by passing the same file twice — SNR should be
`inf dB`, confusions near 0%, genre label unchanged (so the verdict is FAIL):

```bash
python3 run.py sounds/sound.wav sounds/sound.wav
```

Run the unit tests (small synthetic signals and fake models, no model
downloads needed beyond the installed packages), the linter and the type
checker, as CI does on every pull request:

```bash
python3 -m pytest --cov=tester/metrics --cov-fail-under=80
ruff check .
mypy
```

The metrics also work on tensors directly, without files or models:

```python
from tester.metrics import snr, lyrics, genre
snr.snr_db(original, perturbed)          # equal-shape tensors -> dB
snr.measure(original, perturbed, sr)     # windowed + segmental SnrResult
snr.align(original, sr, perturbed, sr2)  # match rate, length, channels
lyrics.confusion("text a", "text b")     # word edit distance in [0, 1]
genre.confusion(embedding_a, embedding_b)  # cosine distance in [0, 2]
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

### Reproducibility

Scores only compare across runs when the stack is identical, so it is pinned:

- `requirements.txt` is a full lock (torch included) compiled from
  `requirements.in`; the regenerate command is at the top of `requirements.in`.
  Install with `pip install -r requirements.txt` only.
- Whisper and MERT load at fixed Hub commits (`WHISPER_REVISION`,
  `MERT_REVISION` in `tester/config.py`). MAEST's weights are fixed by the
  MAEST commit pinned in `requirements.in`.
- Whisper decodes with a fixed language, `temperature=0.0` and
  `condition_on_previous_text=False` (also in `config.py`).
- `results.json["run"]` records the package versions and model revisions used.

---

## Project layout

```
music-perturbation-tester/
├── run.py                 # CLI entry point
├── test.py                # dependency sanity check
├── setup.sh               # full environment installer
├── cleanup.sh             # remove venv + model caches
├── requirements.in        # direct dependencies, exact pins (edit this)
├── requirements.txt       # full lock compiled from requirements.in
├── pyproject.toml         # ruff, mypy, pytest and coverage settings
├── .github/workflows/     # CI: ruff, mypy, pytest
├── tests/                 # unit tests on synthetic signals
├── README.md              # this file
└── tester/                # package
    ├── __init__.py
    ├── config.py          # constants & thresholds
    ├── audio.py           # load / mono / resample / trim
    ├── models.py          # lazy model singletons
    ├── runner.py          # orchestrates the pipeline, one helper per step
    ├── result.py          # RunResult: results.json schema and verdict
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
