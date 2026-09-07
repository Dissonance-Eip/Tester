"""
Pipeline orchestration.

``run(original, perturbed)`` is the single public entry point — it loads
every model, runs every metric, and returns the assembled results.
Side effects (console output, JSON file) are delegated to ``report``.
"""

from typing import Any, Dict

from . import models
from .config import (
    GENRE_CONFUSION_GOOD,
    LYRICS_CONFUSION_GOOD,
    SNR_GOOD,
    SNR_WARN,
)
from .metrics import genre, lyrics, snr
from .report import print_summary, save_json


def run(original_path: str, perturbed_path: str) -> Dict[str, Any]:
    """Evaluate a perturbation against the original audio.

    Runs three metrics:

    1. **SNR**             — how audible the perturbation is to humans
    2. **Lyrics confusion** — does Whisper hear different words?
    3. **Genre + encoder**  — does MAEST relabel? does MERT see a different track?

    All metrics use identical settings on both files so the comparison
    is fair.

    Args:
        original_path:  Path to the clean reference audio.
        perturbed_path: Path to the poisoned audio.

    Returns:
        A structured results dictionary. Also writes ``results.json`` and
        prints a summary to stdout as side effects.
    """
    print("\nLoading models...")
    models.load_all()
    print(f"Evaluating poison: {original_path}  →  {perturbed_path}\n")

    # ── 1. SNR (human audibility) ────────────────────────────
    print("── Step 1/3: Human perceptual quality (SNR) ────")
    snr_db = snr.compute(original_path, perturbed_path)
    snr_label = _snr_label(snr_db)
    print(f"  SNR: {snr_db:.1f} dB  ({snr_label})\n")

    # ── 2. Lyrics confusion (Whisper) ────────────────────────
    print("── Step 2/3: AI lyrics confusion (Whisper) ─────")
    orig_lyrics = lyrics.transcribe(original_path)
    pert_lyrics = lyrics.transcribe(perturbed_path)
    lyric_conf  = lyrics.confusion(orig_lyrics, pert_lyrics)
    print(f"  Original:  {orig_lyrics[:120]}")
    print(f"  Perturbed: {pert_lyrics[:120]}")
    print(f"  Confusion: {lyric_conf:.2%}\n")

    # ── 3. Genre classification + encoder confusion ──────────
    print("── Step 3/3: AI genre classification + encoder ─")
    print("  Classifying original (MAEST)...")
    orig_genres = genre.classify(original_path)
    print("  Classifying perturbed (MAEST)...")
    pert_genres = genre.classify(perturbed_path)

    print("  Original  genres:", _format_genres(orig_genres))
    print("  Perturbed genres:", _format_genres(pert_genres))

    print("\n  Computing encoder confusion (MERT)...")
    fp_orig    = genre.fingerprint(original_path)
    fp_pert    = genre.fingerprint(perturbed_path)
    genre_conf = genre.confusion(fp_orig, fp_pert)
    print(f"  Encoder confusion: {genre_conf:.2%}\n")

    # ── Assemble result ──────────────────────────────────────
    results = _assemble_results(
        original_path  = original_path,
        perturbed_path = perturbed_path,
        snr_db         = snr_db,
        snr_label      = snr_label,
        orig_lyrics    = orig_lyrics,
        pert_lyrics    = pert_lyrics,
        lyric_conf     = lyric_conf,
        orig_genres    = orig_genres,
        pert_genres    = pert_genres,
        genre_conf     = genre_conf,
    )

    print_summary(results)
    save_json(results, "results.json")
    return results


# ── Helpers ──────────────────────────────────────────────────
def _snr_label(snr_db: float) -> str:
    """Return a human-readable label for an SNR value."""
    if snr_db >= SNR_GOOD:
        return "imperceptible"
    if snr_db >= SNR_WARN:
        return "borderline"
    return "noticeable"


def _format_genres(genres) -> str:
    """Pretty-print genre tuples for the console."""
    return "  ".join(f"{g} ({s:.0%})" for g, s in genres)


def _assemble_results(**kw) -> Dict[str, Any]:
    """Build the structured results dictionary saved to ``results.json``.

    Kept here (and not in ``report``) so the runner remains the single
    source of truth for the result schema.
    """
    import datetime
    import os

    from .config import ENCODER_MODEL, MAEST_ARCH, DEVICE, WHISPER_SIZE

    snr_ok      = kw["snr_db"]      >= SNR_GOOD
    lyric_ok    = kw["lyric_conf"]  >= LYRICS_CONFUSION_GOOD
    genre_ok    = kw["genre_conf"]  >= GENRE_CONFUSION_GOOD
    label_changed = kw["orig_genres"][0][0] != kw["pert_genres"][0][0]
    overall_pass  = snr_ok and lyric_ok and genre_ok

    return {
        "_about": (
            "Audio poisoning effectiveness report. A successful perturbation is "
            "inaudible to humans (high SNR) and confuses AI on both lyrics and genre."
        ),
        "run": {
            "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
            "original_file":  os.path.abspath(kw["original_path"]),
            "perturbed_file": os.path.abspath(kw["perturbed_path"]),
            "device":         DEVICE,
            "models": {
                "lyrics_transcriber": WHISPER_SIZE,
                "genre_classifier":   MAEST_ARCH,
                "music_encoder":      ENCODER_MODEL,
            },
        },
        "verdict": {
            "overall_pass":   overall_pass,
            "summary": (
                "Inaudible to humans, confusing to AI."
                if overall_pass
                else "See individual metrics below."
            ),
            "metrics_passed": int(snr_ok) + int(lyric_ok) + int(genre_ok),
            "metrics_total":  3,
        },
        "human_perception": {
            "_description": "How audible the perturbation is to a human listener.",
            "snr_db":       round(kw["snr_db"], 2) if kw["snr_db"] != float("inf") else "inf",
            "label":        kw["snr_label"],
            "threshold":    {"good_above_db": SNR_GOOD, "warn_below_db": SNR_WARN},
            "pass":         snr_ok,
            "interpretation": (
                "Imperceptible to a human listener."        if kw["snr_db"] >= SNR_GOOD else
                "Borderline — some listeners may notice."   if kw["snr_db"] >= SNR_WARN else
                "Clearly audible artifacts — humans hear the perturbation."
            ),
        },
        "ai_lyrics": {
            "_description":     "Does Whisper hear different words on the perturbed file?",
            "confusion_score":  round(kw["lyric_conf"], 4),
            "threshold":        {"good_above": LYRICS_CONFUSION_GOOD},
            "pass":             lyric_ok,
            "original_lyrics":  kw["orig_lyrics"],
            "perturbed_lyrics": kw["pert_lyrics"],
            "interpretation": (
                "Whisper transcribes significantly different lyrics."
                if lyric_ok
                else "Whisper transcribes the same lyrics."
            ),
        },
        "ai_genre": {
            "_description": "Does AI genre classification and music encoding change?",
            "label_classifier": {
                "original_top_genres":  [
                    {"genre": g, "confidence": round(s, 4)} for g, s in kw["orig_genres"]
                ],
                "perturbed_top_genres": [
                    {"genre": g, "confidence": round(s, 4)} for g, s in kw["pert_genres"]
                ],
                "original_top":   kw["orig_genres"][0][0],
                "perturbed_top":  kw["pert_genres"][0][0],
                "label_changed":  label_changed,
            },
            "encoder_confusion": {
                "_description": "Cosine distance between MERT embeddings.",
                "score":        round(kw["genre_conf"], 4),
                "threshold":    {"good_above": GENRE_CONFUSION_GOOD},
                "pass":         genre_ok,
            },
            "interpretation": (
                "AI music encoder sees a meaningfully different track."
                if genre_ok
                else "AI music encoder sees essentially the same track."
            ),
        },
    }
