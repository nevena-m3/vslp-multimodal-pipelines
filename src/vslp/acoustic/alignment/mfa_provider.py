"""Pinned, corpus-level MFA CLI adapter. No model download or runtime G2P."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable
import ast
from uuid import uuid4

import pandas as pd
from .mfa_launcher import LauncherError, MfaLauncher


SUPPORTED = {"3.3.4": {"legacy"}, "3.4.0": {"legacy", "hf_bundle"}}


@dataclass(frozen=True)
class MfaProfile:
    profile_id: str
    mode: str
    acoustic_model: str
    dictionary: str
    language: str = "en"
    phone_set: str = "ARPABET_CMU_39"
    working_sample_rate_hz: int = 16000
    speaker_policy: str = "explicit_deidentified_mapping"
    normalization_profile_id: str = "vslp_prompt_nfkc_v1"
    oov_policy: str = "fail"
    output_format: str = "long_textgrid"
    status: str = "engineering_not_empirically_validated"
    acoustic_model_sha256: str = ""
    dictionary_sha256: str = ""
    acoustic_model_version: str = ""
    dictionary_version: str = ""
    launcher_type: str = "CURRENT_PATH"
    conda_environment: str = ""
    conda_executable: str = ""
    mfa_executable: str = ""
    supported_mfa_version: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "MfaProfile":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        profile = cls(**raw)
        if (profile.mode not in {"legacy", "hf_bundle"}
                or profile.phone_set != "ARPABET_CMU_39"
                or profile.working_sample_rate_hz != 16000
                or profile.speaker_policy != "explicit_deidentified_mapping"
                or profile.normalization_profile_id != "vslp_prompt_nfkc_v1"
                or profile.oov_policy != "fail"
                or profile.output_format != "long_textgrid"
                or profile.status != "engineering_not_empirically_validated"
                or profile.launcher_type.upper() not in {"CONDA_ENV", "EXPLICIT_EXECUTABLE", "CURRENT_PATH"}
                or not profile.acoustic_model or not profile.dictionary):
            raise ValueError("invalid_mfa_profile")
        return profile


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dictionary_oovs(dictionary: Path, transcripts: list[str]) -> list[str]:
    """Conservative local lexicon check; runtime G2P is never enabled."""
    vocabulary = set()
    for line in dictionary.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if fields:
            vocabulary.add(fields[0].casefold())
    words = {word.casefold() for text in transcripts for word in text.split()}
    return sorted(words - vocabulary)


def validate_corpus_pairs(corpus: Path) -> list[str]:
    issues = []
    audio = {path.with_suffix("") for path in corpus.rglob("*.wav")}
    labels = {path.with_suffix("") for path in corpus.rglob("*.lab")}
    if not audio:
        issues.append("empty_corpus")
    issues.extend(f"missing_transcript:{path.name}" for path in sorted(audio - labels))
    issues.extend(f"missing_audio:{path.name}" for path in sorted(labels - audio))
    return issues


def parse_textgrid(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read MFA long TextGrid tiers, omitting blank/silence intervals."""
    text = path.read_text(encoding="utf-8-sig")
    tier = ""
    entries: dict[str, list[dict]] = {"words": [], "phones": []}
    item: dict[str, object] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("name ="):
            tier = stripped.split("=", 1)[1].strip().strip('"').lower()
        elif re.match(r"intervals \[\d+\]:", stripped):
            item = {}
        elif tier in entries and stripped.startswith(("xmin =", "xmax =", "text =")):
            key, value = (part.strip() for part in stripped.split("=", 1))
            item[key] = value.strip('"') if key == "text" else float(value)
            if (key == "text" and str(item[key]).strip()
                    and str(item[key]).strip().casefold() not in {"sp", "sil", "spn"}
                    and "xmin" in item and "xmax" in item):
                entries[tier].append({"label": item["text"], "start_sec": item["xmin"],
                                      "end_sec": item["xmax"]})
    return pd.DataFrame(entries["words"]), pd.DataFrame(entries["phones"])


class MfaProvider:
    name = "mfa"

    def __init__(self, executable: str | None = None,
                 runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> None:
        self.executable = executable
        self.runner = runner
        self.version = ""
        self.mode = ""
        self.commands: list[list[str]] = []
        self.launcher: MfaLauncher | None = None
        self.command_prefix: list[str] = []
        self.resolved_dictionary = ""

    def _configure(self, profile: MfaProfile | None = None) -> dict | None:
        profile = profile or MfaProfile("developer", "legacy", "unused", "unused")
        launcher_type = "EXPLICIT_EXECUTABLE" if self.executable else profile.launcher_type
        self.launcher = MfaLauncher(launcher_type, profile.conda_environment,
                                    profile.conda_executable,
                                    self.executable or profile.mfa_executable, self.runner)
        try:
            self.command_prefix = self.launcher.resolve()
        except (LauncherError, OSError, ValueError, json.JSONDecodeError) as exc:
            return {"status": str(exc), "platform": platform.platform()}
        return None

    def _mfa(self, *args: str) -> list[str]:
        return [*self.command_prefix, *args]

    @staticmethod
    def _temporary_workspace(logs: Path, command: str) -> Path:
        # Kaldi/sqlite tools in MFA 3.3.4 do not reliably handle deep Windows
        # project paths. The run ID remains in the short, isolated temp path.
        run_id = logs.parent.name
        destination = (Path(tempfile.gettempdir()) / "vslp_mfa_work" / run_id /
                       command / uuid4().hex[:8])
        destination.mkdir(parents=True, exist_ok=False)
        return destination

    def detect(self, profile: MfaProfile | None = None) -> dict:
        issue = self._configure(profile)
        if issue:
            return issue
        result = self.runner(self._mfa("version"), capture_output=True, text=True,
                             encoding="utf-8", errors="replace", check=False, timeout=30,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        raw = (result.stdout or result.stderr or "").strip()
        match = re.search(r"(?<!\d)(3\.\d+\.\d+)(?!\d)", raw)
        version = match.group(1) if match else ""
        self.version = version
        return {"status": "AVAILABLE" if version in SUPPORTED else "MFA_VERSION_UNSUPPORTED",
                "executable": self.launcher.resolved_mfa, "version": version,
                "raw_version": raw[:200], "platform": platform.platform(),
                "launcher_type": self.launcher.launcher_type,
                "conda_executable": self.launcher.resolved_conda,
                "conda_environment": self.launcher.conda_environment,
                "environment_path": self.launcher.environment_path,
                "supported_modes": sorted(SUPPORTED.get(version, set()))}

    def inspect_environment(self, profile: MfaProfile) -> dict:
        environment = self.detect(profile)
        if environment["status"] != "AVAILABLE":
            return environment
        if profile.supported_mfa_version and self.version != profile.supported_mfa_version:
            return {**environment, "status": "MFA_VERSION_UNSUPPORTED",
                    "expected_version": profile.supported_mfa_version}
        if profile.mode not in SUPPORTED[self.version]:
            return {**environment, "status": "MFA_VERSION_UNSUPPORTED"}
        model = Path(profile.acoustic_model)
        dictionary = Path(profile.dictionary)
        resources = {}
        for kind, configured in (("acoustic", profile.acoustic_model),
                                 ("dictionary", profile.dictionary)):
            if Path(configured).exists():
                resources[kind] = {"identity": configured, "path": str(Path(configured).resolve())}
                continue
            probe = self.runner(self._mfa("model", "list", kind), capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                check=False, timeout=120,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                names = ast.literal_eval((probe.stdout or "").strip())
            except (ValueError, SyntaxError):
                names = []
            if probe.returncode or configured not in names:
                return {**environment, "status": "MISSING_MODEL" if kind == "acoustic"
                        else "MISSING_DICTIONARY"}
            roots = [Path(root) for root in (os.environ.get("MFA_ROOT_DIR", ""),
                                             str(Path.home() / "Documents" / "MFA")) if root]
            suffix = ".zip" if kind == "acoustic" else ".dict"
            resolved = next((root / "pretrained_models" / kind / f"{configured}{suffix}"
                             for root in roots if (root / "pretrained_models" / kind /
                                                   f"{configured}{suffix}").is_file()), None)
            resources[kind] = {"identity": configured,
                               "path": str(resolved) if resolved else "", "registered": True}
        if profile.mode == "hf_bundle" and not model.is_dir():
            return {**environment, "status": "MISSING_MODEL"}
        if profile.mode == "hf_bundle" and not model.is_dir():
            return {**environment, "status": "MISSING_MODEL"}
        model_path = Path(resources["acoustic"]["path"])
        dictionary_path = Path(resources["dictionary"]["path"])
        model_hash = file_sha256(model_path) if model_path.is_file() else ""
        dictionary_hash = file_sha256(dictionary_path) if dictionary_path.is_file() else ""
        if ((profile.acoustic_model_sha256 and profile.acoustic_model_sha256 != model_hash)
                or (profile.dictionary_sha256 and profile.dictionary_sha256 != dictionary_hash)):
            return {**environment, "status": "RESOURCE_HASH_MISMATCH"}
        if profile.mode != "legacy":
            return {**environment, "status": "HF_PREFLIGHT_NOT_SUPPORTED", "mode": profile.mode}
        commands = {}
        for name in ("validate", "align"):
            probe = self.runner(self._mfa(name, "--help"),
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", check=False, timeout=30,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            commands[name] = probe.returncode == 0
        if not all(commands.values()):
            return {**environment, "status": "MFA_COMMAND_UNAVAILABLE", "commands": commands}
        self.mode = profile.mode
        self.resolved_dictionary = resources["dictionary"]["path"]
        return {**environment, "mode": profile.mode, "model_sha256": model_hash,
                "dictionary_sha256": dictionary_hash, "profile_id": profile.profile_id,
                "commands": commands, "resources": resources}

    def validation_command(self, corpus: Path, profile: MfaProfile) -> list[str]:
        # HF bundles have no validated corpus-validation CLI contract in the pinned
        # versions. Fail closed instead of pretending the legacy validator covers them.
        if profile.mode != "legacy":
            raise RuntimeError("HF_PREFLIGHT_NOT_SUPPORTED")
        return self._mfa("validate", str(corpus), profile.dictionary,
                         "--acoustic_model_path", profile.acoustic_model)

    def alignment_command(self, corpus: Path, output: Path, profile: MfaProfile) -> list[str]:
        if profile.mode == "legacy":
            command = self._mfa("align", str(corpus), profile.dictionary,
                                profile.acoustic_model, str(output))
        else:
            command = self._mfa("align_hf", str(corpus), profile.acoustic_model, str(output))
        return command + ["--output_format", profile.output_format]

    def preflight(self, corpus: Path, profile: MfaProfile, transcripts: list[str],
                  logs: Path) -> dict:
        logs.mkdir(parents=True, exist_ok=True)
        report = self.inspect_environment(profile)
        if report["status"] != "AVAILABLE":
            return report
        pairing_issues = validate_corpus_pairs(corpus)
        if pairing_issues:
            return {**report, "status": "PREFLIGHT_FAILED", "pairing_issues": pairing_issues}
        if not self.resolved_dictionary:
            return {**report, "status": "DICTIONARY_LEXICON_NOT_FOUND"}
        oovs = dictionary_oovs(Path(self.resolved_dictionary), transcripts) if self.resolved_dictionary else []
        report["oov_words"] = oovs
        if oovs:
            return {**report, "status": "OOV"}
        if profile.mode != "legacy":
            return {**report, "status": "HF_PREFLIGHT_NOT_SUPPORTED"}
        temporary = self._temporary_workspace(logs, "validate")
        command = self.validation_command(corpus, profile) + [
            "--temporary_directory", str(temporary), "--clean"]
        self.commands.append(command)
        result = self.runner(command, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=False, timeout=3600,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (logs / "mfa_validation.log").write_text(
            (result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")
        report["validation_command"] = command
        report["validation_temporary_directory"] = str(temporary)
        if result.returncode == 0:
            shutil.rmtree(temporary, ignore_errors=True)
        return {**report, "status": "AVAILABLE" if result.returncode == 0 else "PREFLIGHT_FAILED"}

    def align_corpus(self, corpus: Path, output: Path, profile: MfaProfile,
                     logs: Path) -> dict[str, Path]:
        output.mkdir(parents=True, exist_ok=True)
        temporary = self._temporary_workspace(logs, "align")
        command = self.alignment_command(corpus, output, profile) + [
            "--temporary_directory", str(temporary), "--clean"]
        self.commands.append(command)
        result = self.runner(command, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=False, timeout=14400,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "mfa_alignment.log").write_text(
            (result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")
        if result.returncode:
            raise RuntimeError("ALIGNMENT_FAILED")
        shutil.rmtree(temporary, ignore_errors=True)
        return {path.stem: path for path in output.rglob("*.TextGrid")}

    @staticmethod
    def collect_diagnostics(output: Path, logs: Path) -> list[str]:
        copied = []
        for name in ("alignment_analysis.csv", "unaligned.txt", "oovs_found.txt"):
            for source in output.rglob(name):
                target = logs / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(str(target))
        return copied

    @staticmethod
    def parse_outputs(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        return parse_textgrid(path)
