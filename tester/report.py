"""
Output formatting.

The runner builds a results dict; this module is responsible for turning
it into something a human (the terminal summary) and a machine
(``results.json``) can read.
"""

import json
from typing import Any, Dict

# Decorative widths
_WIDTH = 54


def print_summary(results: Dict[str, Any]) -> None:
    """Print the final pass/fail box to stdout."""
    snr_block   = results["human_perception"]
    lyric_block = results["ai_lyrics"]
    genre_block = results["ai_genre"]

    snr_db    = snr_block["snr_db"]
    snr_ok    = snr_block["pass"]
    snr_warn  = snr_block["threshold"]["warn_below_db"]

    lyric_pct = lyric_block["confusion_score"]
    lyric_ok  = lyric_block["pass"]

    label_orig = genre_block["label_classifier"]["original_top"]
    label_pert = genre_block["label_classifier"]["perturbed_top"]
    label_changed = genre_block["label_classifier"]["label_changed"]
    enc_score = genre_block["encoder_confusion"]["score"]
    enc_ok    = genre_block["encoder_confusion"]["pass"]

    print("\n" + "═" * _WIDTH)
    print("  POISONING EFFECTIVENESS")
    print("═" * _WIDTH)
    print(f"  Human quality  (SNR):    {snr_db} dB   "
          f"{'✅ imperceptible' if snr_ok else '⚠️  borderline' if isinstance(snr_db, (int, float)) and snr_db >= snr_warn else '❌ audible'}")
    print(f"  Lyrics confused:         {lyric_pct:.2%}      "
          f"{'✅' if lyric_ok else '❌ AI still reads lyrics correctly'}")
    print(f"  Genre label:             {label_orig} → {label_pert}   "
          f"{'✅ changed' if label_changed else '❌ unchanged'}")
    print(f"  Encoder confusion:       {enc_score:.2%}      "
          f"{'✅' if enc_ok else '❌ AI still identifies genre'}")
    print("═" * _WIDTH + "\n")


def save_json(results: Dict[str, Any], path: str) -> None:
    """Write the results dict to ``path`` as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Full results saved to {path}")
