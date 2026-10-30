"""Verdict: every README criterion counts, and the console prints it (issue #2)."""

import json
import math

from tester.config import SNR_GOOD
from tester.report import print_summary
from tester.runner import _assemble_results

PASSING = dict(
    original_path="orig.wav",
    perturbed_path="pert.wav",
    snr_db=40.0,
    snr_segmental_min_db=35.0,
    snr_segmental_mean_db=38.0,
    snr_original_silent=False,
    snr_label="imperceptible",
    orig_lyrics="a b c",
    pert_lyrics="x y z",
    lyric_conf=1.0,
    orig_genres=[("Rock---Punk", 0.9)],
    pert_genres=[("Jazz---Bop", 0.8)],
    genre_conf=0.5,
)


def _results(**overrides):
    return _assemble_results(**{**PASSING, **overrides})


def test_all_criteria_met_passes():
    verdict = _results()["verdict"]

    assert verdict["overall_pass"] is True
    assert verdict["metrics_passed"] == verdict["metrics_total"] == 4


def test_unchanged_genre_label_fails_the_verdict():
    verdict = _results(pert_genres=[("Rock---Punk", 0.7)])["verdict"]

    assert verdict["overall_pass"] is False
    assert verdict["metrics_passed"] == 3
    assert verdict["metrics_total"] == 4


def test_snr_gate_uses_the_segmental_minimum():
    results = _results(snr_db=40.0, snr_segmental_min_db=SNR_GOOD - 5)

    assert results["human_perception"]["pass"] is False
    assert results["verdict"]["overall_pass"] is False


def test_infinite_values_serialise_as_valid_json():
    results = _results(
        snr_db=-math.inf,
        snr_segmental_min_db=-math.inf,
        snr_segmental_mean_db=-math.inf,
        snr_original_silent=True,
    )

    text = json.dumps(results, allow_nan=False)

    assert results["human_perception"]["pass"] is False
    assert '"-inf"' in text


def test_console_prints_verdict_and_metrics_passed(capsys):
    print_summary(_results(pert_genres=[("Rock---Punk", 0.7)]))

    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "3/4" in out
