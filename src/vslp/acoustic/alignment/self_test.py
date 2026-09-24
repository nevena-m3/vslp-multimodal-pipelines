"""Isolated, real MFA engineering self-test. No project data is touched."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import monotonic
from typing import Callable
from uuid import uuid4

import pandas as pd
import soundfile as sf

from .mfa_provider import MfaProfile, MfaProvider
from .stage import AlignmentConfig, freeze_alignment, list_alignment_runs, load_final_alignment, run_acoustic_alignment


TEST_TEXT = "We are testing speech today."
TEST_WORDS = ("We", "are", "testing", "speech", "today")


@dataclass
class MfaSelfTestResult:
    environment_status: str = "NOT_RUN"
    mfa_version: str = ""
    provider_mode: str = ""
    acoustic_model: str = ""
    dictionary: str = ""
    fixture_source: str = ""
    fixture_sample_rate: int = 0
    oov_count: int = 0
    mfa_validate_status: str = "NOT_RUN"
    mfa_align_status: str = "NOT_RUN"
    word_count: int = 0
    phone_count: int = 0
    structural_validation_status: str = "NOT_RUN"
    elapsed_time_sec: float = 0.0
    result: str = "FAIL"
    failure_code: str = ""
    failure_message: str = ""
    failure_step: str = ""
    diagnostic_log_path: str = ""
    result_path: str = ""


def generate_windows_speech(destination: Path, text: str = TEST_TEXT) -> None:
    """Use offline System.Speech; never fetch a voice or run ffmpeg."""
    if sys.platform != "win32":
        raise RuntimeError("TTS_UNAVAILABLE")
    escaped_path = str(destination).replace("'", "''")
    escaped_text = text.replace("'", "''")
    scripts = [
        ("$ErrorActionPreference='Stop'; "
         "Add-Type -AssemblyName System.Speech; "
         "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
         f"$s.SetOutputToWaveFile('{escaped_path}'); "
         f"$s.Speak('{escaped_text}'); $s.Dispose()"),
        ("$ErrorActionPreference='Stop'; "
         "$s=New-Object -ComObject SAPI.SpVoice; "
         "$stream=New-Object -ComObject SAPI.SpFileStream; "
         f"$stream.Open('{escaped_path}',3,$false); "
         "$s.AudioOutputStream=$stream; "
         f"$s.Speak('{escaped_text}') | Out-Null; $stream.Close()"),
    ]
    for script in scripts:
        destination.unlink(missing_ok=True)
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                                 "-EncodedCommand", encoded], capture_output=True, text=True,
                                check=False, timeout=90,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode or not destination.is_file():
            continue
        try:
            if sf.info(destination).duration > 0:
                return
        except RuntimeError:
            continue
    destination.unlink(missing_ok=True)
    raise RuntimeError("TTS_UNAVAILABLE")


def _fixture(root: Path, wav: Path) -> tuple[Path, Path]:
    """Create the same upstream input contract used by an ordinary run."""
    (root / "project_manifest.json").write_text(json.dumps({
        "task_id": "mfa_self_test", "task_name": "Nonclinical MFA self-test",
        "task_type": "Other / custom", "run_id": "engineering_self_test"}), encoding="utf-8")
    prompt = root / "prompt.json"
    prompt.write_text(json.dumps({
        "manifest_version": "self_test_1", "task_id": "mfa_self_test",
        "prompt_version": "self_test_1", "language": "en", "transcript": TEST_TEXT,
        "expected_words": TEST_WORDS, "phone_set": "ARPABET_CMU_39"}), encoding="utf-8")
    speakers = root / "speakers.json"
    speakers.write_text(json.dumps({"manifest_version": "self_test_1", "recordings": [
        {"recording_id": "recording_test_01", "speaker_id": "speaker_test_01"}]}),
        encoding="utf-8")
    final = root / "acoustic" / "003_segmentation_review" / "final"
    final.mkdir(parents=True)
    duration = sf.info(wav).duration
    pd.DataFrame([{
        "recording_id": "recording_test_01", "file_name": wav.name,
        "final_decision": "KEEP_MANUAL", "analysis_wav_path": str(wav),
        "analysis_start_sec": 0.0, "analysis_end_sec": duration,
        "segmentation_run_id": "self_test_segmentation", "review_run_id": "self_test_review",
    }]).to_csv(final / "final_segmentation_decisions.csv", index=False)
    pd.DataFrame([{
        "recording_id": "recording_test_01", "view": "authoritative",
        "segment_role": "speech", "start_sec": 0.0, "end_sec": duration,
    }]).to_csv(final / "final_segmentation_intervals.csv", index=False)
    return prompt, speakers


def _retain_failure_logs(root: Path) -> str:
    logs = list((root / "acoustic" / "004_alignment" / "runs").glob("*/logs/*"))
    keep = [path for path in logs if path.is_file() and path.suffix in {".log", ".json"}]
    if not keep:
        return ""
    destination = Path(tempfile.gettempdir()) / "vslp_mfa_self_test_diagnostics" / uuid4().hex
    destination.mkdir(parents=True)
    for path in keep:
        shutil.copy2(path, destination / path.name)
    return str(destination)


def run_mfa_self_test(profile_path: str | Path, fallback_wav_path: str | Path | None = None,
                      *, provider_factory: Callable[[], MfaProvider] = MfaProvider,
                      fixture_generator: Callable[[Path, str], None] = generate_windows_speech,
                      retain_failure_diagnostics: bool = True) -> MfaSelfTestResult:
    """Run actual provider methods by default; injectable dependencies serve unit tests."""
    started = monotonic()
    outcome = MfaSelfTestResult()
    root = Path(tempfile.mkdtemp(prefix="vslp_mfa_self_test_"))
    try:
        outcome.failure_step = "Environment"
        profile = MfaProfile.load(profile_path)
        outcome.acoustic_model = Path(profile.acoustic_model).name
        outcome.dictionary = Path(profile.dictionary).name
        outcome.provider_mode = profile.mode
        provider = provider_factory()
        environment = provider.inspect_environment(profile)
        outcome.environment_status = environment["status"]
        outcome.mfa_version = environment.get("version", "")
        if environment["status"] != "AVAILABLE":
            raise RuntimeError(environment["status"])

        outcome.failure_step = "Fixture generation"
        wav = root / "recording_test_01.wav"
        if fallback_wav_path:
            source = Path(fallback_wav_path)
            if not source.is_file():
                raise RuntimeError("FALLBACK_AUDIO_MISSING")
            shutil.copy2(source, wav)
            outcome.fixture_source = "user_selected_nonclinical_wav"
        else:
            fixture_generator(wav, TEST_TEXT)
            outcome.fixture_source = "windows_offline_system_speech"
        outcome.fixture_sample_rate = sf.info(wav).samplerate
        prompt, speakers = _fixture(root, wav)

        outcome.failure_step = "MFA validation"
        result = run_acoustic_alignment(root, AlignmentConfig(
            source="mfa", prompt_manifest_path=str(prompt),
            speaker_manifest_path=str(speakers), mfa_profile_path=str(profile_path),
            provider=provider))
        run_id = list_alignment_runs(root)[-1]["alignment_run_id"]
        run_dir = root / "acoustic" / "004_alignment" / "runs" / run_id
        preflight = json.loads((run_dir / "logs" / "mfa_preflight.json").read_text(encoding="utf-8"))
        outcome.oov_count = len(preflight.get("oov_words", []))
        outcome.mfa_validate_status = "PASS"
        outcome.mfa_align_status = "PASS" if any("align" in command for command in provider.commands) else "FAIL"
        if result.status != "completed":
            raise RuntimeError("SELF_TEST_ALIGNMENT_INCOMPLETE")

        outcome.failure_step = "Structural validation"
        freeze_alignment(root, run_id)
        store, issue = load_final_alignment(root)
        if issue or store is None:
            raise RuntimeError(issue or "SELF_TEST_RELOAD_FAILED")
        outcome.word_count, outcome.phone_count = len(store.words), len(store.phones)
        if outcome.word_count < 4 or outcome.phone_count == 0:
            raise RuntimeError("SELF_TEST_INSUFFICIENT_TOKENS")
        outcome.structural_validation_status = "PASS"
        outcome.result = "PASS"
        outcome.failure_step = ""
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        outcome.failure_code = str(exc).split(":", 1)[0] or type(exc).__name__
        messages = {
            "MFA_NOT_INSTALLED": "MFA executable could not be found.",
            "CONDA_NOT_FOUND": "Conda could not be located.",
            "MFA_ENVIRONMENT_NOT_FOUND": "The configured MFA Conda environment does not exist.",
            "MFA_NOT_INSTALLED_IN_ENVIRONMENT": "MFA is missing from the configured Conda environment.",
            "MFA_VERSION_UNSUPPORTED": "The installed MFA version is unsupported.",
            "MISSING_MODEL": "The configured acoustic model cannot be resolved.",
            "MISSING_DICTIONARY": "The configured dictionary cannot be resolved.",
            "DICTIONARY_LEXICON_NOT_FOUND": "The installed dictionary file cannot be located for OOV validation.",
            "MFA_COMMAND_UNAVAILABLE": "The required MFA validate/align command is unavailable.",
            "TTS_UNAVAILABLE": "Offline Windows speech generation is unavailable; select a nonclinical WAV.",
            "OOV": "Test transcript words are absent from the configured dictionary.",
            "PREFLIGHT_FAILED": "MFA corpus validation failed; inspect the retained diagnostics.",
            "ALIGNMENT_FAILED": "MFA corpus alignment failed; inspect the retained diagnostics.",
        }
        outcome.failure_message = messages.get(outcome.failure_code, str(exc)[:300])
        if outcome.failure_step == "MFA validation":
            state_files = list((root / "acoustic" / "004_alignment" / "runs").glob("*/logs/mfa_preflight.json"))
            if state_files:
                report = json.loads(state_files[-1].read_text(encoding="utf-8"))
                outcome.oov_count = len(report.get("oov_words", []))
                if outcome.failure_code == "OOV":
                    outcome.failure_message = (
                        f"{outcome.oov_count} test transcript word(s) are not in the dictionary.")
                if report.get("status") == "AVAILABLE":
                    outcome.mfa_validate_status = "PASS"
                    outcome.failure_step = "Corpus alignment"
                else:
                    outcome.mfa_validate_status = "FAIL"
        if retain_failure_diagnostics:
            outcome.diagnostic_log_path = _retain_failure_logs(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        outcome.elapsed_time_sec = round(monotonic() - started, 3)
        results = Path(tempfile.gettempdir()) / "vslp_mfa_self_test_results"
        results.mkdir(parents=True, exist_ok=True)
        result_path = results / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.json"
        outcome.result_path = str(result_path)
        result_path.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an isolated real MFA engineering self-test")
    from .profiles import default_profile_path
    parser.add_argument("--profile", default=os.environ.get("VSLP_MFA_PROFILE", default_profile_path()))
    parser.add_argument("--nonclinical-wav", default="")
    args = parser.parse_args()
    if not args.profile:
        parser.error("--profile or VSLP_MFA_PROFILE is required")
    outcome = run_mfa_self_test(args.profile, args.nonclinical_wav or None)
    print(json.dumps(asdict(outcome), indent=2))
    return 0 if outcome.result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
