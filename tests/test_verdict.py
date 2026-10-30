"""Verdict: every README criterion counts, and the console prints it (issues #2, #3)."""

import dataclasses
import json
import math

import pytest

from tester.config import MERT_REVISION, SNR_GOOD, SNR_WARN, WHISPER_REVISION
from tester.metrics.snr import SnrResult
from tester.report import print_summary
from tester.result import RunResult, snr_label

PASSING = RunResult(
    original_path="orig.wav",
    perturbed_path="pert.wav",
    snr=SnrResult(snr_db=40.0, segmental_min_db=35.0, segmental_mean_db=38.0,
                  original_silent=False),
    original_lyrics="a b c",
    perturbed_lyrics="x y z",
    lyrics_confusion=1.0,
    original_genres=[("Rock---Punk", 0.9)],
    perturbed_genres=[("Jazz---Bop", 0.8)],
    encoder_confusion=0.5,
)


def _results(**overrides):
    return dataclasses.replace(PASSING, **overrides).to_dict()


def _snr(**overrides):
    return PASSING.snr._replace(**overrides)


def test_all_criteria_met_passes():
    verdict = _results()["verdict"]

    assert verdict["overall_pass"] is True
    assert verdict["metrics_passed"] == verdict["metrics_total"] == 4


def test_unchanged_genre_label_fails_the_verdict():
    verdict = _results(perturbed_genres=[("Rock---Punk", 0.7)])["verdict"]

    assert verdict["overall_pass"] is False
    assert verdict["checks"]["genre_label"] is False
    assert verdict["metrics_passed"] == 3
    assert verdict["metrics_total"] == 4


@pytest.mark.parametrize(
    ("overrides", "failed_check"),
    [
        ({"lyrics_confusion": 0.1}, "lyrics"),
        ({"encoder_confusion": 0.1}, "encoder"),
    ],
)
def test_each_ai_gate_can_fail_alone(overrides, failed_check):
    verdict = _results(**overrides)["verdict"]

    assert verdict["checks"][failed_check] is False
    assert verdict["metrics_passed"] == 3


def test_snr_gate_uses_the_segmental_minimum():
    results = _results(snr=_snr(snr_db=40.0, segmental_min_db=SNR_GOOD - 5))

    assert results["human_perception"]["pass"] is False
    assert results["verdict"]["overall_pass"] is False


@pytest.mark.parametrize(
    ("db", "label"),
    [
        (math.inf, "imperceptible"),
        (SNR_GOOD, "imperceptible"),
        (SNR_GOOD - 0.1, "borderline"),
        (SNR_WARN, "borderline"),
        (SNR_WARN - 0.1, "noticeable"),
        (-math.inf, "noticeable"),
    ],
)
def test_snr_label_ladder(db, label):
    assert snr_label(db) == label


def test_label_and_interpretation_follow_the_segmental_minimum():
    section = _results(snr=_snr(segmental_min_db=SNR_WARN + 1))["human_perception"]

    assert section["label"] == "borderline"
    assert section["interpretation"].startswith("Borderline")


def test_infinite_values_serialise_as_valid_json():
    results = _results(snr=SnrResult(-math.inf, -math.inf, -math.inf, original_silent=True))

    text = json.dumps(results, allow_nan=False)

    assert results["human_perception"]["pass"] is False
    assert results["human_perception"]["interpretation"].startswith("Original is silent")
    assert '"-inf"' in text


def test_empty_genre_list_does_not_crash():
    results = _results(original_genres=[], perturbed_genres=[])

    assert results["ai_genre"]["label_classifier"]["original_top"] is None
    assert results["verdict"]["checks"]["genre_label"] is False


def test_console_prints_verdict_and_metrics_passed(capsys):
    print_summary(_results(perturbed_genres=[("Rock---Punk", 0.7)]))

    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "3/4" in out


def test_run_records_versions_and_model_revisions():
    run = _results()["run"]

    assert run["model_revisions"]["music_encoder"] == MERT_REVISION
    assert run["model_revisions"]["lyrics_transcriber"] == WHISPER_REVISION
    assert "genre_classifier" in run["model_revisions"]
    for package in ("python", "torch", "transformers", "faster-whisper", "maest"):
        assert run["versions"][package]
