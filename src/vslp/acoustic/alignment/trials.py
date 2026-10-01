"""Explicit trial boundaries for task-first linguistic alignment.

Reviewed speech intervals are offered as candidates, never silently equated
with repetitions. All coordinates remain on the original recording clock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TRIAL_PROPOSAL_POLICY_VERSION = "reviewed_speech_exact_protocol_count_v1"


def reviewed_trial_candidates(root: str | Path, recording_id: str) -> list[tuple[float, float]]:
    path = (Path(root) / "acoustic" / "003_segmentation_review" / "final" /
            "final_segmentation_intervals.csv")
    if not path.is_file():
        return []
    table = pd.read_csv(path, keep_default_na=False)
    if not {"recording_id", "segment_role", "start_sec", "end_sec"} <= set(table):
        return []
    speech = table.loc[table.recording_id.astype(str).eq(str(recording_id))
                       & table.segment_role.eq("speech")]
    return sorted((float(row.start_sec), float(row.end_sec)) for row in speech.itertuples())


def structurally_proposed_trials(root: str | Path, context: dict) -> dict:
    """Propose only unambiguous reviewed speech regions; never invent trials."""
    choices = context["choices"]
    task = context.get("task") or {}
    if (not task.get("alignment_applicable") or choices.get("pending_trial_revision")
            or choices.get("manual_trial_review_required")):
        return choices
    final = Path(root) / "acoustic" / "003_segmentation_review" / "final"
    interval_path = final / "final_segmentation_intervals.csv"
    decision_path = final / "final_segmentation_decisions.csv"
    if not interval_path.is_file() or not decision_path.is_file():
        return choices
    intervals = pd.read_csv(interval_path, keep_default_na=False)
    decisions = pd.read_csv(decision_path, keep_default_na=False)
    if not {"recording_id", "segment_role", "start_sec", "end_sec"} <= set(intervals):
        return choices
    reviewed_bounds = {str(row.recording_id): (float(row.analysis_start_sec),
                                               float(row.analysis_end_sec))
                       for row in decisions.itertuples()}
    by_record = {str(identity): frame for identity, frame in intervals.groupby("recording_id")}
    prompts = {item["prompt_id"]: item for item in task.get("prompts", [])}
    for record in context.get("records", []):
        identity = str(record["recording_id"])
        if choices.get("trials", {}).get(identity):
            continue
        prompt_id = record.get("prompt_id") or (
            next(iter(prompts)) if len(prompts) == 1 else "")
        prompt = prompts.get(prompt_id)
        if not prompt or prompt.get("expected_repetitions") is None:
            continue
        frame = by_record.get(identity)
        if frame is None or identity not in reviewed_bounds:
            continue
        candidates = sorted((float(row.start_sec), float(row.end_sec))
                            for row in frame.loc[frame.segment_role.eq("speech")].itertuples())
        if len(candidates) != int(prompt["expected_repetitions"]):
            continue
        excluded = [(float(row.start_sec), float(row.end_sec))
                    for row in frame.loc[frame.segment_role.eq("manual_exclusion")].itertuples()]
        start_bound, end_bound = reviewed_bounds[identity]
        previous_end = start_bound
        valid = True
        for start, end in candidates:
            if (not start_bound <= previous_end <= start < end <= end_bound or
                    any(left < end and right > start for left, right in excluded)):
                valid = False
                break
            previous_end = end
        if not valid:
            continue
        choices.setdefault("trials", {})[identity] = [
            {"trial_id": f"trial_{number:03d}", "start_sec": start,
             "end_sec": end, "prompt_id": prompt_id,
             "proposal_source": "AUTO_STRUCTURAL_CANDIDATE",
             "review_policy_version": TRIAL_PROPOSAL_POLICY_VERSION}
            for number, (start, end) in enumerate(candidates, 1)]
    return choices


def validate_trial_draft(root: str | Path, recording_id: str,
                         regions: list[tuple[float, float]]) -> None:
    """Keep every trial on the frozen reviewed clock and outside exclusions."""
    folder = Path(root) / "acoustic" / "003_segmentation_review" / "final"
    decisions = pd.read_csv(folder / "final_segmentation_decisions.csv",
                            keep_default_na=False)
    chosen = decisions.loc[decisions.recording_id.astype(str).eq(str(recording_id))]
    if len(chosen) != 1:
        raise ValueError(f"MISSING_REVIEWED_RECORDING:{recording_id}")
    row = chosen.iloc[0]
    intervals = pd.read_csv(folder / "final_segmentation_intervals.csv",
                            keep_default_na=False)
    excluded = intervals.loc[intervals.recording_id.astype(str).eq(str(recording_id))
                             & intervals.segment_role.eq("manual_exclusion")]
    previous_end = float(row.analysis_start_sec)
    for start, end in sorted(regions):
        if not (float(row.analysis_start_sec) <= previous_end <= start < end <=
                float(row.analysis_end_sec)):
            raise ValueError(f"INVALID_TRIAL_BOUNDARIES:{recording_id}")
        if any(float(item.start_sec) < end and float(item.end_sec) > start
               for item in excluded.itertuples()):
            raise ValueError(f"TRIAL_CROSSES_EXCLUSION:{recording_id}")
        previous_end = end


def validate_trial_choices(root: str | Path, context: dict) -> list[dict]:
    choices = context["choices"].get("trials", {})
    for record in context["records"]:
        if not choices.get(record["recording_id"]):
            raise ValueError(f"TRIAL_BOUNDARIES_REQUIRED:{record['recording_id']}")
    interval_path = (Path(root) / "acoustic" / "003_segmentation_review" / "final" /
                     "final_segmentation_intervals.csv")
    intervals = pd.read_csv(interval_path, keep_default_na=False)
    decisions = pd.read_csv(interval_path.with_name("final_segmentation_decisions.csv"),
                            keep_default_na=False).set_index("recording_id")
    result = []
    for record in context["records"]:
        identity = record["recording_id"]
        entries = choices.get(identity, [])
        bounds = decisions.loc[identity]
        previous_end = float(bounds.analysis_start_sec)
        for number, entry in enumerate(sorted(entries, key=lambda item: float(item["start_sec"])), 1):
            start, end = float(entry["start_sec"]), float(entry["end_sec"])
            if not float(bounds.analysis_start_sec) <= previous_end <= start < end <= float(bounds.analysis_end_sec):
                raise ValueError(f"INVALID_TRIAL_BOUNDARIES:{identity}")
            excluded = intervals.loc[intervals.recording_id.astype(str).eq(identity)
                                     & intervals.segment_role.eq("manual_exclusion")]
            if any(float(row.start_sec) < end and float(row.end_sec) > start
                   for row in excluded.itertuples()):
                raise ValueError(f"TRIAL_CROSSES_EXCLUSION:{identity}")
            if context["task"]["alignment_applicable"]:
                prompt_id = (entry.get("prompt_id") or record["prompt_id"] or
                             context["task"]["prompts"][0]["prompt_id"])
                if prompt_id != (record["prompt_id"] or
                                 context["task"]["prompts"][0]["prompt_id"]):
                    raise ValueError(f"TRIAL_PROMPT_MISMATCH:{identity}")
            else:
                prompt_id = ""
            result.append({"recording_id": identity,
                           "trial_id": str(entry.get("trial_id") or f"trial_{number:03d}"),
                           "prompt_id": prompt_id, "start_sec": start, "end_sec": end,
                           "proposal_source": entry.get("proposal_source", "HUMAN_CONFIRMED"),
                           "review_policy_version": entry.get("review_policy_version", "")})
            previous_end = end
    return result


def trial_manifest_for_alignment(root: str | Path, context: dict) -> Path:
    root = Path(root)
    trials = validate_trial_choices(root, context)
    path = root / "configs" / "alignment_task" / "alignment_trials.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "1", "source": "versioned_reviewed_trial_proposals",
                                "trials": trials}, indent=2), encoding="utf-8")
    return path


def confirmed_task_trials(root: str | Path, recording_id: str) -> list[dict]:
    """Read explicitly confirmed, project-local trials for any task type."""
    path = Path(root) / "configs" / "alignment_choices.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw.get("trials", {}).get(str(recording_id), []))


def trial_identity_for_interval(root: str | Path, recording_id: str,
                                start_sec: float, end_sec: float,
                                trials: list[dict] | None = None) -> str:
    matches = [str(item["trial_id"]) for item in (
        trials if trials is not None else confirmed_task_trials(root, recording_id))
               if float(item["start_sec"]) - 1e-7 <= start_sec < end_sec <=
               float(item["end_sec"]) + 1e-7]
    return matches[0] if len(matches) == 1 else ""
