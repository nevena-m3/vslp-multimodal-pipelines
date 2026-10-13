"""Nominal Bamboo WPM from the frozen prompt and F07 reviewed timing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vslp.acoustic.alignment.reviewed_trials import load_confirmed_trials
from vslp.acoustic.alignment.task_workflow import task_entry
from vslp.acoustic.features.family07_v2 import reviewed_timing
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


VERSION = "family08-bamboo-nominal-wpm-2.0.0"
FAMILY08_IDS = frozenset({"bamboo_speaking_rate_wpm", "bamboo_articulation_rate_wpm"})
PROMPT_SHA256 = "a132fa58fe945d86812dec3cce3936f6f586a24527105f3bc9009f18efdb5f67"


def frozen_bamboo_prompt() -> dict:
    """Verify the registered scientific count without using MFA tokenization."""
    entry = task_entry("bamboo_passage")
    prompt = entry["prompts"][0] if entry and entry.get("prompts") else {}
    text = str(prompt.get("exact_expected_text", ""))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if (prompt.get("prompt_id") != "bamboo_passage_v1"
            or prompt.get("expected_repetitions") != 1
            or prompt.get("scientific_word_count") != 97
            or prompt.get("prompt_sha256") != PROMPT_SHA256
            or digest != PROMPT_SHA256
            or len(text.split()) != 97 or "good-looking" not in text):
        raise ValueError("invalid_prompt_word_count")
    return prompt


def nominal_rates(timing: dict[str, tuple[float, str]],
                  word_count: int | None = None) -> dict[str, tuple[float, str]]:
    """Calculate leaves independently from the exact F07 denominator values."""
    if word_count is None:
        word_count = int(frozen_bamboo_prompt()["scientific_word_count"])
    ids = {"bamboo_speaking_rate_wpm": ("bamboo_total_duration_s", "missing_bamboo_total_duration",
                                         "total_duration_nonpositive", "nonfinite_speaking_rate"),
           "bamboo_articulation_rate_wpm": ("bamboo_speech_duration_s", "missing_bamboo_speech_duration",
                                             "speech_duration_nonpositive", "nonfinite_articulation_rate")}
    result = {}
    for feature, (denominator, missing, nonpositive, nonfinite) in ids.items():
        value, reason = timing.get(denominator, (np.nan, missing))
        value = float(value)
        if not isinstance(word_count, int) or word_count <= 0:
            result[feature] = (np.nan, "invalid_prompt_word_count")
        elif reason:
            result[feature] = (np.nan, reason)
        elif not np.isfinite(value):
            result[feature] = (np.nan, missing)
        elif value <= 0:
            result[feature] = (np.nan, nonpositive)
        else:
            rate = 60.0 * word_count / value
            result[feature] = ((rate, "") if np.isfinite(rate) else (np.nan, nonfinite))
    total = float(timing.get("bamboo_total_duration_s", (np.nan, ""))[0])
    speech = float(timing.get("bamboo_speech_duration_s", (np.nan, ""))[0])
    if np.isfinite([total, speech]).all() and total > 0 and speech > 0:
        if speech > total + 1e-8:
            return {feature: (np.nan, "speech_duration_exceeds_total") for feature in result}
        speaking, articulation = (result["bamboo_speaking_rate_wpm"],
                                  result["bamboo_articulation_rate_wpm"])
        if not speaking[1] and not articulation[1] and articulation[0] + 1e-8 < speaking[0]:
            return {feature: (np.nan, "rate_consistency_failure") for feature in result}
    return result


def _trial_timeline(intervals: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    """Clip reviewed intervals to one human-confirmed trial without joining gaps."""
    frame = intervals.copy()
    frame["start_sec"] = frame.start_sec.astype(float).clip(lower=start)
    frame["end_sec"] = frame.end_sec.astype(float).clip(upper=end)
    return frame.loc[frame.end_sec.gt(frame.start_sec)].copy()


def bamboo_trial_rates(root: Path, recording_id: str, intervals: pd.DataFrame,
                       word_count: int | None = None) -> list[dict]:
    """Use independently reviewed trials where present; otherwise one passage domain."""
    choices = root / "configs" / "alignment_choices.json"
    if choices.is_file():
        trials, issue = load_confirmed_trials(root, recording_id, "bamboo_passage")
        if issue:
            return [{"trial_id": "", "sequence_id": "", "failure_reason": issue}]
    else:
        speech = intervals.loc[intervals.segment_role.astype(str).eq("speech")]
        if speech.empty:
            return [{"trial_id": "", "sequence_id": "", "failure_reason":
                     "missing_bamboo_speech_duration"}]
        trials = [{"trial_id": f"{recording_id}:passage_001",
                   "sequence_id": f"{recording_id}:passage_001",
                   "start_sec": float(speech.start_sec.min()),
                   "end_sec": float(speech.end_sec.max()),
                   "trial_review_source": "FINAL_REVIEWED_SEGMENTATION"}]
    rows = []
    for trial in trials:
        start, end = float(trial["start_sec"]), float(trial["end_sec"])
        timing, _ = reviewed_timing(_trial_timeline(intervals, start, end), "bamboo_passage")
        rates = nominal_rates(timing, word_count)
        rows.append({"trial_id": str(trial["trial_id"]), "sequence_id": str(trial["sequence_id"]),
                     "trial_start_sec": start, "trial_end_sec": end,
                     "trial_review_source": trial.get("trial_review_source", ""),
                     "bamboo_total_duration_s": timing.get("bamboo_total_duration_s", (np.nan, ""))[0],
                     "bamboo_speech_duration_s": timing.get("bamboo_speech_duration_s", (np.nan, ""))[0],
                     **{feature: value for feature, (value, _) in rates.items()},
                     **{f"{feature}_reason": reason for feature, (_, reason) in rates.items()}})
    return rows


def run_family08_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (intervals_csv is None or Path(intervals_csv).resolve() != intervals.resolve()
            or Path(decisions_csv).resolve() != decisions.resolve()):
        raise ValueError("Family 08 requires frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    timeline = pd.read_csv(intervals, keep_default_na=False)
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_id = str(project.get("task_id", ""))
    selected = set(config.selected_features) & FAMILY08_IDS
    specs = registry.set_index("feature")
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables = stage / "tables"
    native = tables / "native_measurements"
    values_rows, status_rows, trial_rows = [], [], []
    try:
        prompt = frozen_bamboo_prompt()
        prompt_issue = ""
    except ValueError as exc:
        prompt, prompt_issue = {}, str(exc)
    for done, record in enumerate(kept.itertuples(), 1):
        recording_id = str(record.recording_id)
        own = timeline.loc[timeline.recording_id.astype(str).eq(recording_id)].copy()
        own = own.loc[own.view.astype(str).eq("authoritative")] if "view" in own else own
        trials = (bamboo_trial_rates(root, recording_id, own,
                                     int(prompt["scientific_word_count"]))
                  if task_id == "bamboo_passage"
                  and not prompt_issue else [])
        for trial in trials:
            trial_rows.append({"recording_id": recording_id, "task_id": task_id,
                               "prompt_id": prompt.get("prompt_id", ""),
                               "prompt_version": prompt.get("prompt_version", ""),
                               "prompt_sha256": prompt.get("prompt_sha256", ""),
                               "expected_word_count": prompt.get("scientific_word_count", np.nan),
                               "word_count_source": "expected_prompt_count",
                               "prompt_completion_verified": False,
                               "denominator_source": "F07_FINAL_REVIEWED_TIMING",
                               "qc_status": "unavailable" if trial.get("failure_reason")
                               or any(trial.get(f"{feature}_reason") for feature in FAMILY08_IDS)
                               else "computed",
                               "qc_flags": ";".join(filter(None, [trial.get("failure_reason", ""),
                                   *(trial.get(f"{feature}_reason", "") for feature in FAMILY08_IDS)])),
                               **trial})
        base = {"recording_id": recording_id, "file_name": str(record.file_name),
                "task_id": task_id, "prompt_id": prompt.get("prompt_id", ""),
                "prompt_version": prompt.get("prompt_version", ""),
                "prompt_sha256": prompt.get("prompt_sha256", ""),
                "expected_word_count": prompt.get("scientific_word_count", np.nan),
                "word_count_source": "expected_prompt_count",
                "prompt_completion_verified": False,
                "numerator_source": "expected_prompt_count",
                "denominator_source": "F07_FINAL_REVIEWED_TIMING",
                "segmentation_review_hash": sha256_file(decisions) + ":" + sha256_file(intervals),
                "algorithm_version": VERSION}
        value_row = dict(base)
        for feature in sorted(selected):
            valid = [float(trial[feature]) for trial in trials
                     if feature in trial and np.isfinite(trial[feature])
                     and not trial.get(f"{feature}_reason")]
            reason = ("task_not_applicable" if task_id != "bamboo_passage" else prompt_issue
                      or ("" if valid else next((trial.get(f"{feature}_reason", "")
                                                 or trial.get("failure_reason", "") for trial in trials),
                                                "missing_reviewed_segmentation")))
            value = float(np.median(valid)) if valid else np.nan
            value_row[feature] = value
            status_rows.append({**base, "feature": feature, "feature_id": feature,
                                "family_id": "F08", "value": value,
                                "status": "computed" if not reason else "unavailable",
                                "failure_reason": reason, "unit": str(specs.loc[feature, "unit"]),
                                "input_contract": "FINAL_REVIEWED_TIMING",
                                "trial_count": len(trials), "valid_trial_count": len(valid)})
        values_rows.append(value_row)
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {record.file_name}")
    native.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    trial_path = native / "family08_bamboo_trial_rates.csv"
    values_df, status_df = pd.DataFrame(values_rows), pd.DataFrame(status_rows)
    values_df.to_csv(values_path, index=False)
    status_df.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, values_df, registry, status_df,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff)
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=VERSION,
        status="completed_with_warnings" if status_df.failure_reason.astype(str).ne("").any()
        else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(trial_path), "family08_bamboo_trial_rates", "text/csv")],
        config={"selected_features": sorted(selected), "prompt_sha256":
                prompt.get("prompt_sha256", ""), "expected_word_count":
                prompt.get("scientific_word_count", np.nan),
                "word_count_source": "expected_prompt_count",
                "prompt_completion_verified": False,
                "input_contract": "FINAL_REVIEWED_TIMING"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
