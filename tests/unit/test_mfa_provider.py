"""Pure and mocked corpus-level MFA adapter tests; no MFA installation required."""

from __future__ import annotations

import subprocess

from vslp.acoustic.alignment.mfa_provider import (
    MfaProfile, MfaProvider, dictionary_oovs, validate_corpus_pairs,
)
from vslp.acoustic.alignment.stage import normalize_transcript


def _profile(tmp_path, mode="legacy"):
    model = tmp_path / "model.zip"
    model.write_bytes(b"model")
    dictionary = tmp_path / "lexicon.dict"
    dictionary.write_text("ONE W AH N\nTWO T UW\nTHREE TH R IY\n", encoding="utf-8")
    return MfaProfile("engineering_v1", mode, str(model), str(dictionary))


def test_version_gate_and_command_modes(tmp_path):
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "MFA 3.4.0", "")

    provider = MfaProvider(executable="C:/mfa.exe", runner=runner)
    profile = _profile(tmp_path)
    assert provider.inspect_environment(profile)["status"] == "AVAILABLE"
    command = provider.alignment_command(tmp_path / "corpus", tmp_path / "out", profile)
    assert command[1] == "align"
    assert "align_one" not in command
    hf = MfaProfile("hf", "hf_bundle", str(tmp_path), profile.dictionary)
    assert provider.alignment_command(tmp_path / "corpus", tmp_path / "out", hf)[1] == "align_hf"
    assert provider.validation_command(tmp_path / "corpus", profile)[1] == "validate"
    assert calls[0][1] == "version"


def test_unsupported_version_and_oov_stop_preflight(tmp_path):
    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, "MFA 4.0.0", "")

    provider = MfaProvider(executable="C:/mfa.exe", runner=runner)
    profile = _profile(tmp_path)
    assert provider.inspect_environment(profile)["status"] == "MFA_VERSION_UNSUPPORTED"
    assert dictionary_oovs(tmp_path / "lexicon.dict", ["ONE UNKNOWN"]) == ["unknown"]


def test_normalization_is_deterministic_and_rejects_numbers():
    assert normalize_transcript("  co-operate, don’t!  ") == "CO OPERATE DON'T"
    try:
        normalize_transcript("I have 2 words")
    except ValueError as exc:
        assert str(exc) == "TRANSCRIPT_NUMBER_REQUIRES_REVIEW"
    else:
        raise AssertionError("numeral was silently normalized")


def test_corpus_pairing_and_preflight_oov(tmp_path):
    corpus = tmp_path / "corpus" / "speaker_A"
    corpus.mkdir(parents=True)
    (corpus / "r1.wav").write_bytes(b"synthetic-placeholder")
    assert validate_corpus_pairs(corpus.parent) == ["missing_transcript:r1"]
    (corpus / "r1.lab").write_text("ONE UNKNOWN")
    assert validate_corpus_pairs(corpus.parent) == []

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, "MFA 3.3.4", "")

    provider = MfaProvider(executable="C:/mfa.exe", runner=runner)
    report = provider.preflight(corpus.parent, _profile(tmp_path), ["ONE UNKNOWN"], tmp_path / "logs")
    assert report["status"] == "OOV"
    assert report["oov_words"] == ["unknown"]


def test_provider_preserves_raw_analysis_metrics_without_quality_threshold(tmp_path):
    output = tmp_path / "provider"
    output.mkdir()
    raw = output / "alignment_analysis.csv"
    raw.write_text("file,overall_log_likelihood,speech_log_likelihood,phone_duration_deviation,snr\n"
                   "r1,-900,-20,6,3\n", encoding="utf-8")
    paths = MfaProvider.collect_diagnostics(output, tmp_path / "logs")
    assert len(paths) == 1
    assert (tmp_path / "logs" / "alignment_analysis.csv").read_bytes() == raw.read_bytes()


def test_every_validate_and_align_uses_distinct_run_local_temp(tmp_path):
    profile = _profile(tmp_path)
    corpus = tmp_path / "corpus" / "speaker_A"
    corpus.mkdir(parents=True)
    (corpus / "r1.wav").write_bytes(b"test")
    (corpus / "r1.lab").write_text("ONE", encoding="utf-8")
    stale_global = tmp_path / "Documents" / "MFA" / "mfa_input" / "phones.txt"
    stale_global.parent.mkdir(parents=True)
    stale_global.write_text("stale", encoding="utf-8")
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "MFA 3.3.4", "")

    provider = MfaProvider(executable="C:/mfa.exe", runner=runner)
    logs = tmp_path / "run_A" / "logs"
    assert provider.preflight(corpus.parent, profile, ["ONE"], logs)["status"] == "AVAILABLE"
    provider.align_corpus(corpus.parent, tmp_path / "out_A", profile, logs)
    assert provider.preflight(corpus.parent, profile, ["ONE"], logs)["status"] == "AVAILABLE"
    provider.align_corpus(corpus.parent, tmp_path / "out_B", profile, logs)
    execution = [command for command in calls if "--temporary_directory" in command]
    workspaces = [command[command.index("--temporary_directory") + 1] for command in execution]
    assert len(workspaces) == len(set(workspaces)) == 4
    assert all("vslp_mfa_work" in path and logs.parent.name in path for path in workspaces)
    assert all("--clean" in command for command in execution)
    assert stale_global.read_text(encoding="utf-8") == "stale"
