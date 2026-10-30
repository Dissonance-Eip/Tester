"""run() wiring with every model faked: steps feed RunResult and the report (issue #3)."""

import json

import torch

from tester import runner

from .signals import noise_at_snr, sine


def test_run_writes_the_report(write_wav, tmp_path, monkeypatch, capsys):
    clean = sine(1.0)
    original = write_wav("orig", clean)
    perturbed = write_wav("pert", clean + noise_at_snr(clean, 40.0))
    transcripts = {original: "hello world", perturbed: "goodbye world"}
    genres = {original: [("Rock---Punk", 0.9)], perturbed: [("Jazz---Bop", 0.6)]}
    fingerprints = {original: torch.tensor([1.0, 0.0]), perturbed: torch.tensor([0.0, 1.0])}
    monkeypatch.setattr(runner.models, "load_all", lambda: None)
    monkeypatch.setattr(runner.lyrics, "transcribe", transcripts.__getitem__)
    monkeypatch.setattr(runner.genre, "classify", genres.__getitem__)
    monkeypatch.setattr(runner.genre, "fingerprint", fingerprints.__getitem__)
    output = tmp_path / "report.json"

    results = runner.run(original, perturbed, str(output))

    assert json.loads(output.read_text()) == results
    assert results["ai_lyrics"]["confusion_score"] == 0.5
    assert results["ai_genre"]["encoder_confusion"]["score"] == 1.0
    assert results["ai_genre"]["label_classifier"]["label_changed"] is True
    assert results["human_perception"]["label"] == "imperceptible"
    assert results["verdict"]["overall_pass"] is True
    assert "PASS" in capsys.readouterr().out
