"""Self-test exercises the production provider and stage with a mocked CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

from vslp.acoustic.alignment.mfa_provider import MfaProvider
from vslp.acoustic.alignment import self_test


def _profile(tmp_path: Path, *, omit_today: bool = False) -> Path:
    model = tmp_path / "english_us_arpa.zip"
    model.write_bytes(b"engineering-model-placeholder")
    dictionary = tmp_path / "english_us_arpa.dict"
    words = ["WE", "ARE", "TESTING", "SPEECH"]
    if not omit_today:
        words.append("TODAY")
    dictionary.write_text("\n".join(f"{word} W AH" for word in words), encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"profile_id": "engineering_test", "mode": "legacy",
                                   "acoustic_model": str(model), "dictionary": str(dictionary)}),
                       encoding="utf-8")
    return profile


def _grid(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    lines = ['File type = "ooTextFile"', 'name = "words"']
    for index, word in enumerate(("WE", "ARE", "TESTING", "SPEECH", "TODAY"), 1):
        start = .1 + (index - 1) * .35
        lines += [f"intervals [{index}]:", f" xmin = {start}",
                  f" xmax = {start + .3}", f' text = "{word}"']
    lines.append('name = "phones"')
    for index, phone in enumerate(("W", "AA1", "T", "S", "T"), 1):
        start = .12 + (index - 1) * .35
        lines += [f"intervals [{index}]:", f" xmin = {start}",
                  f" xmax = {start + .1}", f' text = "{phone}"']
    (output / "recording_test_01.TextGrid").write_text("\n".join(lines), encoding="utf-8")


def _provider_factory(calls: list[list[str]]):
    def runner(command, **_kwargs):
        calls.append(command)
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, "MFA 3.3.4", "")
        if command[1] == "align" and "--help" not in command:
            corpus = Path(command[2])
            speaker = corpus / "speaker_test_01"
            assert (speaker / "recording_test_01.wav").is_file()
            assert (speaker / "recording_test_01.lab").read_text() == "WE ARE TESTING SPEECH TODAY"
            _grid(Path(command[5]))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    return lambda: MfaProvider(executable="C:/mfa.exe", runner=runner)


def _fake_tts(path: Path, _text: str) -> None:
    sf.write(path, np.zeros(2 * 16000, dtype="float32"), 16000)


def test_self_test_full_handoff_and_cleanup(tmp_path, monkeypatch):
    case = tmp_path / "case"
    monkeypatch.setattr(self_test.tempfile, "mkdtemp", lambda **_kwargs: str(case))
    monkeypatch.setattr(self_test.tempfile, "gettempdir", lambda: str(tmp_path))
    case.mkdir()
    calls = []
    result = self_test.run_mfa_self_test(_profile(tmp_path),
                                         provider_factory=_provider_factory(calls),
                                         fixture_generator=_fake_tts)
    assert result.result == "PASS", result
    assert result.word_count == 5 and result.phone_count == 5
    assert result.structural_validation_status == "PASS"
    assert result.fixture_source == "windows_offline_system_speech"
    assert sum(command[1] == "validate" and "--help" not in command for command in calls) == 1
    assert sum(command[1] == "align" and "--help" not in command for command in calls) == 1
    assert all(command[1] != "align_one" for command in calls)
    assert not case.exists()
    assert Path(result.result_path).is_file()


def test_self_test_fallback_wav_does_not_invoke_tts(tmp_path, monkeypatch):
    case = tmp_path / "case"
    monkeypatch.setattr(self_test.tempfile, "mkdtemp", lambda **_kwargs: str(case))
    monkeypatch.setattr(self_test.tempfile, "gettempdir", lambda: str(tmp_path))
    case.mkdir()
    fallback = tmp_path / "nonclinical.wav"
    _fake_tts(fallback, "")
    calls = []
    result = self_test.run_mfa_self_test(
        _profile(tmp_path), fallback, provider_factory=_provider_factory(calls),
        fixture_generator=lambda *_: (_ for _ in ()).throw(AssertionError("TTS called")))
    assert result.result == "PASS", result
    assert result.fixture_source == "user_selected_nonclinical_wav"
    assert not case.exists()


def test_oov_fails_before_align_and_cleans_corpus(tmp_path, monkeypatch):
    case = tmp_path / "case"
    monkeypatch.setattr(self_test.tempfile, "mkdtemp", lambda **_kwargs: str(case))
    monkeypatch.setattr(self_test.tempfile, "gettempdir", lambda: str(tmp_path))
    case.mkdir()
    calls = []
    result = self_test.run_mfa_self_test(_profile(tmp_path, omit_today=True),
                                         provider_factory=_provider_factory(calls),
                                         fixture_generator=_fake_tts)
    assert result.result == "FAIL"
    assert result.failure_code == "OOV"
    assert result.oov_count == 1
    assert not any(command[1] == "align" and "--help" not in command for command in calls)
    assert not case.exists()
