"""run.py argument handling: bad input fails fast, before any model loads (issue #3)."""

import pytest

import run as cli
import tester
from tester import config


@pytest.fixture
def audio_pair(write_wav):
    from .signals import sine

    return write_wav("orig", sine(0.1)), write_wav("pert", sine(0.1))


@pytest.fixture
def fake_run(monkeypatch):
    """Record the call instead of loading models; keep env changes local to the test."""
    calls = []
    monkeypatch.setattr(tester, "run", lambda *args: calls.append(args))
    monkeypatch.delenv(config.DEVICE_ENV, raising=False)
    monkeypatch.delenv(config.MAX_SECONDS_ENV, raising=False)
    return calls


def test_missing_file_is_an_error(tmp_path, fake_run, capsys):
    code = cli.main([str(tmp_path / "nope.wav"), str(tmp_path / "nope2.wav")])

    assert code == 2
    assert "original file not found" in capsys.readouterr().err
    assert fake_run == []


def test_non_positive_seconds_is_an_error(audio_pair, fake_run, capsys):
    code = cli.main([*audio_pair, "--seconds", "0"])

    assert code == 2
    assert "--seconds must be a positive integer" in capsys.readouterr().err


def test_missing_output_directory_is_an_error(audio_pair, tmp_path, fake_run, capsys):
    code = cli.main([*audio_pair, "--output", str(tmp_path / "no" / "r.json")])

    assert code == 2
    assert "output directory does not exist" in capsys.readouterr().err


def test_unknown_device_is_rejected(audio_pair, fake_run):
    with pytest.raises(SystemExit):
        cli.main([*audio_pair, "--device", "tpu"])


def test_unavailable_device_is_an_error(audio_pair, fake_run, monkeypatch, capsys):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    code = cli.main([*audio_pair, "--device", "cuda"])

    assert code == 2
    assert "not available" in capsys.readouterr().err


def test_valid_arguments_reach_the_runner(audio_pair, tmp_path, fake_run):
    out = tmp_path / "report.json"

    code = cli.main([*audio_pair, "--output", str(out), "--seconds", "10", "--device", "cpu"])

    assert code == 0
    assert fake_run == [(audio_pair[0], audio_pair[1], str(out))]
    assert cli.os.environ[config.MAX_SECONDS_ENV] == "10"
    assert cli.os.environ[config.DEVICE_ENV] == "cpu"


@pytest.mark.parametrize(("cap", "expected"), [(None, 30), ("10", 10), ("90", 30)])
def test_seconds_cap_only_lowers_windows(monkeypatch, cap, expected):
    if cap is None:
        monkeypatch.delenv(config.MAX_SECONDS_ENV, raising=False)
    else:
        monkeypatch.setenv(config.MAX_SECONDS_ENV, cap)

    assert config._seconds_cap(30) == expected
