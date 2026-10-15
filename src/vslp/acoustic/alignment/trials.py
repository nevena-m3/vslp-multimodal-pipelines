"""Structural trial proposals on the original recording clock.

Reviewed speech intervals cannot distinguish a pause from a restart. Structural
acceptance is workflow triage, never linguistic or boundary validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


TRIAL_PROPOSAL_POLICY_VERSION = "task_specific_structural_review_v3"
TRIAL_STRUCTURAL_LIMIT = (
    "Reviewed speech intervals do not establish linguistic completeness or "
    "distinguish an ordinary within-utterance pause from a restart.")
READY_TRIAL_STATES = {"READY_FOR_ALIGNMENT"}
TRIAL_REVIEW_STATES = {"PROPOSED", "READY_FOR_ALIGNMENT", "NEEDS_TRIAL_REVIEW",
                       "OMITTED", "STALE"}


def normalized_trial_state(entry: dict) -> dict:
    """Read legacy pre-MFA choices without treating post-MFA acceptance as readiness."""
    result = dict(entry)
    if "alignment_review_status" in result:
        raise ValueError("INVALID_PRE_MFA_ALIGNMENT_STATUS")
    if "trial_status" in result:
        if result["trial_status"] not in TRIAL_REVIEW_STATES:
            raise ValueError("INVALID_TRIAL_STATUS")
    else:
        legacy = result.get("review_status", "")
        mapping = {"HUMAN_ACCEPTED": ("READY_FOR_ALIGNMENT", "HUMAN"),
                   "HUMAN_CONFIRMED": ("READY_FOR_ALIGNMENT", "HUMAN"),
                   "PROPOSED": ("PROPOSED", "AUTO"),
                   "NEEDS_REVIEW": ("NEEDS_TRIAL_REVIEW", "AUTO"),
                   "HUMAN_REJECTED": ("OMITTED", "HUMAN"),
                   "OMITTED": ("OMITTED", "HUMAN"),
                   "STALE": ("STALE", "AUTO")}
        if legacy in mapping:
            result["trial_status"], result["trial_review_source"] = mapping[legacy]
        else:
            result["trial_status"] = "STALE"
            result["trial_review_source"] = "LEGACY_AMBIGUOUS"
            result["trial_review_reason"] = ["LEGACY_TRIAL_STATE_REQUIRES_REVIEW"]
    result.setdefault("trial_review_source", "AUTO")
    result.setdefault("trial_review_reason", result.get("review_reason", []))
    result.setdefault("trial_review_timestamp", result.get("review_timestamp_utc", ""))
    result.pop("review_status", None)
    return result


def trial_proposal_review_queue(context: dict) -> list[dict]:
    """Only unresolved trial structure enters the scientist's pre-MFA queue."""
    choices = context.get("choices", {})
    queue = []
    for record in context.get("records", []):
        identity = record["recording_id"]
        entries = choices.get("trials", {}).get(identity, [])
        if not entries:
            queue.append({"recording_id": identity, "trial_id": "",
                          "trial_status": "NEEDS_TRIAL_REVIEW",
                          "trial_review_reason": ["TRIAL_BOUNDARIES_REQUIRED"]})
        for entry in entries:
            current = normalized_trial_state(entry)
            status = current["trial_status"]
            if status in {"PROPOSED", "NEEDS_TRIAL_REVIEW", "STALE"}:
                queue.append({"recording_id": identity, "trial_id": entry["trial_id"],
                              "trial_status": status,
                              "trial_review_reason": current["trial_review_reason"]})
    return queue


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
    """Classify deterministic structure; queue only observed exceptions."""
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
        existing = choices.get("trials", {}).get(identity, [])
        if existing:
            for index, raw_entry in enumerate(existing):
                entry = normalized_trial_state(raw_entry)
                existing[index] = entry
                registered_prompt = prompts.get(entry.get("prompt_id"))
                review_changed = bool(entry.get("segmentation_review_hash") and
                    entry["segmentation_review_hash"] != context.get("segmentation_review_hash"))
                identity_changed = bool(entry.get("task_id") and
                    entry["task_id"] != task["task_id"] or entry.get("prompt_version") and
                    (registered_prompt is None or entry["prompt_version"] !=
                     registered_prompt["prompt_version"]))
                missing_auto_hash = (entry["trial_status"] == "READY_FOR_ALIGNMENT"
                                     and entry.get("trial_review_source") == "AUTO"
                                     and not entry.get("segmentation_review_hash"))
                if (entry["trial_status"] in READY_TRIAL_STATES and
                        (review_changed or identity_changed or missing_auto_hash)):
                    entry["trial_status"] = "STALE"
                    entry["trial_review_source"] = "AUTO"
                    entry["trial_review_reason"] = ["UPSTREAM_REVIEW_OR_PROMPT_CHANGED"]
            continue
        if choices.get("trial_edit_pending", {}).get(identity):
            continue
        prompt_id = record.get("prompt_id") or (
            next(iter(prompts)) if len(prompts) == 1 else "")
        prompt = prompts.get(prompt_id)
        if not prompt or prompt.get("prompt_version") in (None, ""):
            continue
        frame = by_record.get(identity)
        if frame is None or identity not in reviewed_bounds:
            continue
        speech = sorted((float(row.start_sec), float(row.end_sec))
                            for row in frame.loc[frame.segment_role.eq("speech")].itertuples())
        if not speech:
            continue
        excluded = [(float(row.start_sec), float(row.end_sec))
                    for row in frame.loc[frame.segment_role.eq("manual_exclusion")].itertuples()]
        start_bound, end_bound = reviewed_bounds[identity]
        if any(not (math.isfinite(start) and math.isfinite(end) and
                    start_bound <= start < end <= end_bound) for start, end in speech):
            continue
        # Passage pauses belong inside one candidate. Sentence speech spans are
        # possible attempts, but VAD alone cannot establish their identity.
        candidates = ([(speech[0][0], speech[-1][1])]
                      if task.get("task_id") == "bamboo_passage" else speech)
        expected = prompt.get("expected_repetitions")
        flags = []
        blocking_flags = []
        if expected is None:
            flags.append("PROTOCOL_COUNT_NOT_FROZEN")
        elif len(candidates) != int(expected):
            flags.append("REPETITION_COUNT_MISMATCH")
        if task.get("task_id") == "bamboo_passage" and expected != 1:
            flags.append("BAMBOO_PROTOCOL_CONFLICT")
            blocking_flags.append("BAMBOO_PROTOCOL_CONFLICT")
        if record.get("prompt_id") and record["prompt_id"] != prompt_id:
            flags.append("PROMPT_ID_CONFLICT")
            blocking_flags.append("PROMPT_ID_CONFLICT")
        # An explicit upstream attempt label, if present, is evidence of a
        # separate attempt. Ordinary segmentation intervals have no such label.
        if "linguistic_attempt_id" in frame:
            attempt_ids = {str(value).strip() for value in frame.loc[
                frame.segment_role.eq("speech"), "linguistic_attempt_id"]
                if str(value).strip()}
            if len(attempt_ids) > 1:
                flags.append("MULTIPLE_EXPLICIT_ATTEMPTS")
                blocking_flags.append("MULTIPLE_EXPLICIT_ATTEMPTS")
        if any(left < end and right > start for start, end in candidates
               for left, right in excluded):
            flags.append("TRIAL_CROSSES_EXCLUSION")
            blocking_flags.append("TRIAL_CROSSES_EXCLUSION")
        if choices.get("trial_edit_pending", {}).get(identity):
            flags.append("STALE_AFTER_TRIAL_EDIT")
            blocking_flags.append("STALE_AFTER_TRIAL_EDIT")
        status = "NEEDS_TRIAL_REVIEW" if blocking_flags else "READY_FOR_ALIGNMENT"
        choices.setdefault("trials", {})[identity] = [
            {"trial_id": f"trial_{number:03d}", "start_sec": start,
             "end_sec": end, "prompt_id": prompt_id,
             "task_id": task["task_id"], "prompt_version": prompt["prompt_version"],
             "proposal_source": "AUTO_STRUCTURAL_CANDIDATE",
             "trial_status": status, "trial_review_source": "AUTO",
             "trial_review_reason": blocking_flags, "trial_review_timestamp": "",
             "protocol_expected_repetitions": expected,
             "observed_candidate_count": len(candidates),
             "protocol_count_match": expected is not None and len(candidates) == int(expected),
             "flags": flags, "trial_policy_version": TRIAL_PROPOSAL_POLICY_VERSION,
             "segmentation_review_hash": context.get("segmentation_review_hash", ""),
             "structural_limit": TRIAL_STRUCTURAL_LIMIT,
             "proposal_flags": flags,
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
            entry = normalized_trial_state(entry)
            start, end = float(entry["start_sec"]), float(entry["end_sec"])
            if not (math.isfinite(start) and math.isfinite(end) and
                    float(bounds.analysis_start_sec) <= previous_end <= start < end <=
                    float(bounds.analysis_end_sec)):
                raise ValueError(f"INVALID_TRIAL_BOUNDARIES:{identity}")
            speech = intervals.loc[intervals.recording_id.astype(str).eq(identity)
                                   & intervals.segment_role.eq("speech")]
            if not any(float(row.start_sec) < end and float(row.end_sec) > start
                       for row in speech.itertuples()):
                raise ValueError(f"TRIAL_OUTSIDE_REVIEWED_SPEECH:{identity}")
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
                registered_prompt = next((item for item in context["task"]["prompts"]
                                          if item["prompt_id"] == prompt_id), None)
                if (registered_prompt is None or
                        entry.get("task_id", context["task_id"]) != context["task_id"] or
                        entry.get("prompt_version", registered_prompt["prompt_version"]) !=
                        registered_prompt["prompt_version"]):
                    raise ValueError(f"TRIAL_PROMPT_MISMATCH:{identity}")
            else:
                prompt_id = ""
            result.append({"recording_id": identity,
                           "trial_id": str(entry.get("trial_id") or f"trial_{number:03d}"),
                           "task_id": context["task_id"], "prompt_id": prompt_id,
                           "prompt_version": next((item["prompt_version"] for item in
                               context["task"]["prompts"] if item["prompt_id"] == prompt_id), "")
                               if context["task"]["alignment_applicable"] else "",
                           "start_sec": start, "end_sec": end,
                           "proposal_source": entry.get("proposal_source", "HUMAN_CONFIRMED"),
                           "initial_proposal_source": entry.get("initial_proposal_source", ""),
                           "proposal_flags": entry.get("proposal_flags", []),
                           "protocol_expected_repetitions": entry.get("protocol_expected_repetitions"),
                           "observed_candidate_count": entry.get("observed_candidate_count"),
                           "protocol_count_match": entry.get("protocol_count_match"),
                           "flags": entry.get("flags", entry.get("proposal_flags", [])),
                           "trial_policy_version": entry.get("trial_policy_version",
                                                               entry.get("review_policy_version", "")),
                           "trial_status": entry["trial_status"],
                           "trial_review_source": entry["trial_review_source"],
                           "trial_review_reason": entry["trial_review_reason"],
                           "trial_review_timestamp": entry["trial_review_timestamp"],
                           "segmentation_review_hash": entry.get("segmentation_review_hash", ""),
                           "structural_limit": entry.get("structural_limit", TRIAL_STRUCTURAL_LIMIT),
                           "review_policy_version": entry.get("review_policy_version", "")})
            previous_end = end
    return result


def trial_manifest_for_alignment(root: str | Path, context: dict) -> Path:
    root = Path(root)
    if (context.get("issue") or not context.get("segmentation_review_hash") or
            not (context.get("task") or {}).get("alignment_applicable")):
        raise ValueError("ALIGNMENT_CONTEXT_NOT_READY")
    trials = validate_trial_choices(root, context)
    if any(item["segmentation_review_hash"] and item["segmentation_review_hash"] !=
           context.get("segmentation_review_hash") for item in trials):
        raise ValueError("STALE_TRIAL_PROPOSAL:SEGMENTATION_REVIEW_CHANGED")
    if any(context["choices"].get("trial_edit_pending", {}).get(
            item["recording_id"]) for item in trials):
        raise ValueError("TRIAL_CONFIRMATION_REQUIRED:STALE_AFTER_TRIAL_EDIT")
    pending = [item for item in trials if item["trial_status"] not in READY_TRIAL_STATES]
    if pending:
        raise ValueError(f"TRIAL_CONFIRMATION_REQUIRED:{pending[0]['recording_id']}")
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
