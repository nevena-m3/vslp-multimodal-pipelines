"""Resolve Alignment inputs from explicit project task and recording metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pandas as pd

from .mfa_provider import MfaProfile, MfaProvider, dictionary_oovs
from .profiles import default_profile_path
from .stage import AlignmentConfig, normalize_transcript, run_acoustic_alignment


REGISTRY_PATH = Path(__file__).with_name("task_prompts.json")


def task_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def task_entry(task_id: str) -> dict | None:
    return next((task for task in task_registry()["tasks"]
                 if task["task_id"] == str(task_id).casefold()), None)


def project_choices_path(root: Path) -> Path:
    return root / "configs" / "alignment_choices.json"


def load_project_choices(root: Path) -> dict:
    path = project_choices_path(root)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": "1", "recordings": {}}


def save_project_choices(root: Path, choices: dict) -> None:
    path = project_choices_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(choices, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _reviewed_records(root: Path) -> pd.DataFrame:
    path = root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_decisions.csv"
    if not path.is_file():
        return pd.DataFrame()
    table = pd.read_csv(path, keep_default_na=False)
    return table.loc[table.final_decision.isin(["KEEP_AUTO", "KEEP_MANUAL"])].copy()


def _explicit_speakers(root: Path) -> dict[str, str]:
    path = root / "acoustic" / "000_metadata" / "tables" / "project_file_index.csv"
    if not path.is_file():
        return {}
    table = pd.read_csv(path, keep_default_na=False)
    required = {"recording_id", "subject_id", "subject_id_source"}
    if not required <= set(table):
        return {}
    table = table.loc[table.subject_id_source.eq("metadata_csv")]
    return {str(row.recording_id): str(row.subject_id)
            for row in table.itertuples()
            if re.fullmatch(r"[A-Za-z0-9_]+", str(row.subject_id))}


def project_alignment_context(root: str | Path) -> dict:
    root = Path(root)
    manifest_path = root / "project_manifest.json"
    if not manifest_path.is_file():
        return {"issue": "PROJECT_NOT_SELECTED", "records": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_id = str(manifest.get("task_id", ""))
    task = task_entry(task_id)
    records = _reviewed_records(root)
    choices = load_project_choices(root)
    explicit_speakers = _explicit_speakers(root)
    selections = []
    for row in records.itertuples():
        identity = str(row.recording_id)
        stored = choices.get("recordings", {}).get(identity, {})
        selections.append({"recording_id": identity, "file_name": str(row.file_name),
                           "speaker_id": stored.get("speaker_id") or explicit_speakers.get(identity, ""),
                           "prompt_id": stored.get("prompt_id", ""),
                           "alignment_transcript": stored.get("alignment_transcript", ""),
                           "transcript_reason": stored.get("transcript_reason", "")})
    return {"task_id": task_id, "task_name": manifest.get("task_display_name") or
            manifest.get("task_name") or task_id, "task_type": manifest.get("task_type"),
            "task": task, "records": selections,
            "reviewed": bool(records.shape[0]), "choices": choices}


def resolve_prompt(context: dict, prompt_id: str = "") -> tuple[dict | None, str]:
    task = context.get("task")
    if not task:
        return None, "TASK_NOT_REGISTERED"
    if not task["alignment_applicable"]:
        return None, "ALIGNMENT_NOT_APPLICABLE"
    prompts = task["prompts"]
    if not prompts:
        return None, "MISSING_CANONICAL_PROMPT"
    if not prompt_id and len(prompts) == 1:
        return prompts[0], ""
    prompt = next((item for item in prompts if item["prompt_id"] == prompt_id), None)
    return (prompt, "") if prompt else (None, "PROMPT_SELECTION_REQUIRED")


def build_task_alignment_config(root: str | Path, *, profile_path: str = "") -> AlignmentConfig:
    """Persist internal manifests from the selected project; never infer stimuli from names."""
    root = Path(root)
    context = project_alignment_context(root)
    records = context["records"]
    if not records:
        raise ValueError("MISSING_REVIEWED_RECORDINGS")
    prompts_by_recording = {}
    for item in records:
        prompt, issue = resolve_prompt(context, item["prompt_id"])
        if issue:
            raise ValueError(issue)
        prompts_by_recording[item["recording_id"]] = prompt
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", item["speaker_id"])
           for item in records):
        raise ValueError("SPEAKER_ID_REQUIRED")
    folder = root / "configs" / "alignment_task"
    folder.mkdir(parents=True, exist_ok=True)
    prompt_paths = {}
    for identity, prompt in prompts_by_recording.items():
        expected = normalize_transcript(prompt["exact_expected_text"]).split()
        prompt_file = folder / f"prompt_{identity}.json"
        prompt_file.write_text(json.dumps({"manifest_version": task_registry()["registry_version"],
            "task_id": context["task_id"], "prompt_id": prompt["prompt_id"],
            "prompt_version": prompt["prompt_version"], "language": prompt["language"],
            "transcript": prompt["exact_expected_text"], "expected_words": expected,
            "phone_set": "ARPABET_CMU_39"}, indent=2), encoding="utf-8")
        prompt_paths[identity] = str(prompt_file)
    speakers = folder / "speakers.json"
    speakers.write_text(json.dumps({"manifest_version": "1", "recordings": [
        {"recording_id": item["recording_id"], "speaker_id": item["speaker_id"]}
        for item in records]}, indent=2), encoding="utf-8")
    overrides = []
    reviewer = context["choices"].get("reviewer_id", "")
    for item in records:
        text = item["alignment_transcript"].strip()
        expected_text = prompts_by_recording[item["recording_id"]]["exact_expected_text"]
        if text and normalize_transcript(text) != normalize_transcript(expected_text):
            if not item["transcript_reason"]:
                raise ValueError("TRANSCRIPT_OVERRIDE_REASON_REQUIRED")
            if not reviewer:
                raise ValueError("TRANSCRIPT_OVERRIDE_REVIEWER_REQUIRED")
            overrides.append({"recording_id": item["recording_id"],
                "alignment_transcript": text, "reason": item["transcript_reason"],
                "reviewer": reviewer, "reviewed_utc": datetime.now(timezone.utc).isoformat(),
                "prompt_id": prompts_by_recording[item["recording_id"]]["prompt_id"],
                "prompt_version": prompts_by_recording[item["recording_id"]]["prompt_version"]})
    override_path = folder / "transcript_overrides.json"
    if overrides:
        first_prompt = next(iter(prompts_by_recording.values()))
        override_path.write_text(json.dumps({"prompt_version": first_prompt["prompt_version"],
            "overrides": overrides}, indent=2), encoding="utf-8")
    return AlignmentConfig(source="mfa", prompt_manifest_path=next(iter(prompt_paths.values())),
        speaker_manifest_path=str(speakers),
        transcript_overrides_path=str(override_path) if overrides else "",
        mfa_profile_path=profile_path or default_profile_path(),
        recording_prompt_manifest_paths=prompt_paths)


def check_task_preflight(root: str | Path, profile_path: str = "") -> dict:
    context = project_alignment_context(root)
    root = Path(root)
    interval_path = root / "acoustic" / "003_segmentation_review" / "final" / "final_segmentation_intervals.csv"
    if not interval_path.is_file():
        return {"status": "ACTION_REQUIRED", "issue": "MISSING_FROZEN_SEGMENTATION",
                "context": context}
    reviewed = _reviewed_records(root)
    if reviewed.empty or any(not Path(str(path)).is_file()
                             for path in reviewed.analysis_wav_path):
        return {"status": "ACTION_REQUIRED", "issue": "MISSING_REVIEWED_RECORDING",
                "context": context}
    try:
        config = build_task_alignment_config(root, profile_path=profile_path)
    except ValueError as exc:
        return {"status": "ACTION_REQUIRED", "issue": str(exc), "context": context}
    profile = MfaProfile.load(config.mfa_profile_path)
    provider = MfaProvider()
    environment = provider.inspect_environment(profile)
    if environment["status"] != "AVAILABLE":
        return {"status": "ACTION_REQUIRED", "issue": environment["status"],
                "context": context, "environment": environment}
    from .stage import PromptManifest
    prompts = [PromptManifest.load(path) for path in config.recording_prompt_manifest_paths.values()]
    if not provider.resolved_dictionary:
        return {"status": "ACTION_REQUIRED", "issue": "DICTIONARY_LEXICON_NOT_FOUND",
                "context": context, "environment": environment}
    transcripts = [item["alignment_transcript"] or prompt.transcript
                   for item, prompt in zip(context["records"], prompts)]
    oovs = dictionary_oovs(Path(provider.resolved_dictionary),
                           [normalize_transcript(text) for text in transcripts])
    if oovs:
        return {"status": "ACTION_REQUIRED", "issue": "OOV", "oov_words": oovs,
                "context": context, "environment": environment}
    return {"status": "READY", "context": context, "environment": environment,
            "prompt": prompts[0].transcript}


def check_mfa_environment(profile_path: str = "") -> dict:
    profile = MfaProfile.load(profile_path or default_profile_path())
    return MfaProvider().inspect_environment(profile)


def run_task_alignment(output_root: str | Path, profile_path: str = "",
                       progress_callback=None):
    config = build_task_alignment_config(output_root, profile_path=profile_path)
    if progress_callback:
        progress_callback(0, 0, "Preparing corpus...")
    return run_acoustic_alignment(output_root, config, progress_callback=progress_callback)
