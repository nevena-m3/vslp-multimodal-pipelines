"""Resolve Alignment inputs from explicit project task and recording metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

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


def _manifest_task(manifest: dict) -> dict | None:
    """Resolve only explicit Setup metadata, never recording names or audio."""
    registered = task_entry(str(manifest.get("task_id", "")))
    if registered:
        return registered
    names = [str(manifest.get(field, "")).strip().casefold()
             for field in ("task_display_name", "task_name")]
    matches = [task for task in task_registry()["tasks"]
               if task["display_name"].casefold() in names]
    return matches[0] if len(matches) == 1 else None


def project_choices_path(root: Path) -> Path:
    return root / "configs" / "alignment_choices.json"


def load_project_choices(root: Path) -> dict:
    path = project_choices_path(root)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": "1", "recordings": {}}


def save_project_choices(root: Path, choices: dict) -> None:
    path = project_choices_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    from uuid import uuid4
    import time

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(choices, indent=2, ensure_ascii=False), encoding="utf-8")
        for attempt in range(5):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


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


def missing_speaker_recordings(context: dict) -> list[str]:
    """Recordings without an explicit, MFA-safe deidentified speaker identity."""
    return [item["recording_id"] for item in context.get("records", [])
            if not re.fullmatch(r"[A-Za-z0-9_]+", item["speaker_id"])]


def project_alignment_context(root: str | Path) -> dict:
    root = Path(root)
    manifest_path = root / "project_manifest.json"
    if not manifest_path.is_file():
        return {"issue": "PROJECT_NOT_SELECTED", "records": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = _manifest_task(manifest)
    task_id = task["task_id"] if task else str(manifest.get("task_id", ""))
    records = _reviewed_records(root)
    choices = load_project_choices(root)
    explicit_speakers = _explicit_speakers(root)
    selections = []
    changed = False
    for row in records.itertuples():
        identity = str(row.recording_id)
        stored = choices.get("recordings", {}).get(identity, {})
        if not (stored.get("speaker_id") or explicit_speakers.get(identity)
                or stored.get("technical_speaker_key")):
            stored = choices.setdefault("recordings", {}).setdefault(identity, {})
            stored["technical_speaker_key"] = f"technical_{uuid4().hex}"
            changed = True
        speaker = stored.get("speaker_id") or explicit_speakers.get(identity) or stored.get("technical_speaker_key", "")
        selections.append({"recording_id": identity, "file_name": str(row.file_name),
                           "speaker_id": speaker,
                           "speaker_id_source": ("reviewer" if stored.get("speaker_id") else
                                                 "metadata_csv" if explicit_speakers.get(identity) else
                                                 "technical_recording_key"),
                           "prompt_id": stored.get("prompt_id", ""),
                           "alignment_transcript": stored.get("alignment_transcript", ""),
                           "transcript_reason": stored.get("transcript_reason", "")})
    if changed:
        save_project_choices(root, choices)
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


def build_task_prompt_manifests(root: str | Path, context: dict | None = None) -> dict[str, str]:
    """Materialize registered task text for either MFA or validated import."""
    root = Path(root)
    context = context or project_alignment_context(root)
    if not context["records"]:
        raise ValueError("MISSING_REVIEWED_RECORDINGS")
    folder = root / "configs" / "alignment_task"
    prompt_paths = {}
    for item in context["records"]:
        prompt, issue = resolve_prompt(context, item["prompt_id"])
        if issue:
            raise ValueError(issue)
        identity = item["recording_id"]
        expected = normalize_transcript(prompt["exact_expected_text"]).split()
        folder.mkdir(parents=True, exist_ok=True)
        prompt_file = folder / f"prompt_{identity}.json"
        prompt_file.write_text(json.dumps({"manifest_version": task_registry()["registry_version"],
            "task_id": context["task_id"], "prompt_id": prompt["prompt_id"],
            "prompt_version": prompt["prompt_version"], "language": prompt["language"],
            "expected_repetitions": prompt.get("expected_repetitions"),
            "transcript": prompt["exact_expected_text"], "expected_words": expected,
            "phone_set": "ARPABET_CMU_39"}, indent=2), encoding="utf-8")
        prompt_paths[identity] = str(prompt_file)
    return prompt_paths


def build_task_alignment_config(root: str | Path, *, profile_path: str = "") -> AlignmentConfig:
    """Persist internal manifests from the selected project; never infer stimuli from names."""
    root = Path(root)
    context = project_alignment_context(root)
    from .trials import structurally_proposed_trials
    before = json.dumps(context["choices"], sort_keys=True)
    context["choices"] = structurally_proposed_trials(root, context)
    if json.dumps(context["choices"], sort_keys=True) != before:
        save_project_choices(root, context["choices"])
    records = context["records"]
    if not records:
        raise ValueError("MISSING_REVIEWED_RECORDINGS")
    prompts_by_recording = {}
    for item in records:
        prompt, issue = resolve_prompt(context, item["prompt_id"])
        if issue:
            raise ValueError(issue)
        prompts_by_recording[item["recording_id"]] = prompt
    if missing_speaker_recordings(context):
        raise ValueError("SPEAKER_ID_REQUIRED")
    folder = root / "configs" / "alignment_task"
    prompt_paths = build_task_prompt_manifests(root, context)
    from .trials import trial_manifest_for_alignment
    trial_manifest = trial_manifest_for_alignment(root, context)
    speakers = folder / "speakers.json"
    speakers.write_text(json.dumps({"manifest_version": "1", "recordings": [
        {"recording_id": item["recording_id"], "speaker_id": item["speaker_id"],
         "speaker_id_source": item["speaker_id_source"]}
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
                           structural_auto_review=True,
        speaker_manifest_path=str(speakers),
        transcript_overrides_path=str(override_path) if overrides else "",
        mfa_profile_path=profile_path or default_profile_path(),
        trial_manifest_path=str(trial_manifest),
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
    missing_speakers = missing_speaker_recordings(context)
    if missing_speakers:
        return {"status": "ACTION_REQUIRED", "issue": "SPEAKER_ID_REQUIRED",
                "missing_speaker_recordings": missing_speakers, "context": context}
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
    repetition_gaps = {}
    for item in context["records"]:
        expected_prompt, _ = resolve_prompt(context, item["prompt_id"])
        expected = expected_prompt.get("expected_repetitions") if expected_prompt else None
        if expected is not None:
            actual = len(context["choices"].get("trials", {}).get(item["recording_id"], []))
            if actual < int(expected):
                repetition_gaps[item["recording_id"]] = {"expected": int(expected),
                                                          "confirmed": actual}
    return {"status": "READY", "context": context, "environment": environment,
            "prompt": prompts[0].transcript, "repetition_gaps": repetition_gaps}


def check_mfa_environment(profile_path: str = "") -> dict:
    profile = MfaProfile.load(profile_path or default_profile_path())
    return MfaProvider().inspect_environment(profile)


def run_task_alignment(output_root: str | Path, profile_path: str = "",
                       progress_callback=None, rerun_recording_id: str = "",
                       reuse_run_id: str = ""):
    config = build_task_alignment_config(output_root, profile_path=profile_path)
    if rerun_recording_id:
        from dataclasses import replace
        config = replace(config, rerun_recording_id=rerun_recording_id,
                         reuse_run_id=reuse_run_id)
    if progress_callback:
        progress_callback(0, 0, "Preparing corpus...")
    return run_acoustic_alignment(output_root, config, progress_callback=progress_callback)
