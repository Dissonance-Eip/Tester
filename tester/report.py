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
    """Print the final summary box to stdout."""
    snr_block   = results["human_perception"]
    lyric_block = results["ai_lyrics"]
    genre_block = results["ai_genre"]

    snr_db    = snr_block["snr_db"]
    snr_min   = snr_block["segmental_snr_db"]["min"]
    verdict   = results["verdict"]

    lyric_pct = lyric_block["confusion_score"]

    label_orig = genre_block["label_classifier"]["original_top"]
    label_pert = genre_block["label_classifier"]["perturbed_top"]
    enc_score = genre_block["encoder_confusion"]["score"]

    print("\n" + "═" * _WIDTH)
    print("  POISONING EFFECTIVENESS")
    print("═" * _WIDTH)
    print(f"  Human quality  (SNR):    {snr_db} dB  (segmental min {snr_min} dB)")
    print(f"  Lyrics confused:         {lyric_pct:.2%}")
    print(f"  Genre label:             {label_orig} → {label_pert}")
    print(f"  Encoder confusion:       {enc_score:.2%}")
    print("─" * _WIDTH)
    print(f"  Verdict:                 {'PASS' if verdict['overall_pass'] else 'FAIL'}  "
          f"({verdict['metrics_passed']}/{verdict['metrics_total']} metrics passed)")
    print("═" * _WIDTH + "\n")


def save_json(results: Dict[str, Any], path: str) -> None:
    """Write the results dict to ``path`` as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Full results saved to {path}")
