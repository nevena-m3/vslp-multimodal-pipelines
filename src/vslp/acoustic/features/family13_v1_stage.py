"""F13 reviewed-audio executor: canonical Bamboo sentences, never raw MFA output."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import __version__ as SCIPY_VERSION

from vslp.acoustic.alignment import load_final_alignment
from vslp.acoustic.alignment.final_contract import canonical_units
from vslp.acoustic.alignment.reviewed_trials import load_confirmed_trials
from vslp.acoustic.features.catalog import project_task_key
from vslp.acoustic.features.family07_v2 import _eligible_speech
from vslp.acoustic.features.family11_v2_stage import _bounds
from vslp.acoustic.features.family13_v1 import (
    FS, MIN_SENTENCES, PUBLIC_IDS, VERSION, analysis_copy, ppe_from_audio,
    mfcc_trajectory, rhythm_components, rqa_components, specs, tsfel_entropy, visibility_10ms,
    wpd_components,
)
from vslp.acoustic.features.stable_phonation import centered_stable_spans
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _bamboo_sentence_indices() -> list[tuple[int, int]]:
    """Canonical lexical ranges from the frozen prompt, not provider tokens."""
    prompt_path = Path(__file__).parents[1] / "alignment" / "task_prompts.json"
    registry = json.loads(prompt_path.read_text(encoding="utf-8"))
    task = next(t for t in registry["tasks"] if t["task_id"] == "bamboo_passage")
    prompt = next(p for p in task["prompts"] if p["prompt_id"] == "bamboo_passage_v1")
    sentences = re.split(r"(?<=[.!?])\s+", prompt["exact_expected_text"].strip())
    lengths = [len(canonical_units(sentence)) for sentence in sentences]
    if len(lengths) != 8 or sum(lengths) != 97:
        raise ValueError("bamboo_prompt_sentence_contract_invalid")
    left = 1
    result = []
    for length in lengths:
        result.append((left, left + length - 1))
        left += length
    return result


def _sentence_spans(alignment, recording_id: str, trial_id: str) -> list[dict]:
    words = alignment.words_for_trial(recording_id, trial_id)
    if len(words) != 97:
        raise ValueError("bamboo_canonical_coverage_incomplete")
    words = words.sort_values("canonical_word_index")
    if words.canonical_word_index.astype(int).tolist() != list(range(1, 98)):
        raise ValueError("bamboo_canonical_order_invalid")
    prompt_path = Path(__file__).parents[1] / "alignment" / "task_prompts.json"
    registry = json.loads(prompt_path.read_text(encoding="utf-8"))
    task = next(t for t in registry["tasks"] if t["task_id"] == "bamboo_passage")
    prompt = next(p for p in task["prompts"] if p["prompt_id"] == "bamboo_passage_v1")
    expected = [unit.casefold() for unit in canonical_units(prompt["exact_expected_text"])]
    actual = words.canonical_word_text.astype(str).str.casefold().tolist()
    if actual != expected:
        raise ValueError("bamboo_canonical_words_mismatch")
    spans = []
    for sentence_index, (first, last) in enumerate(_bamboo_sentence_indices(), 1):
        selection = words.loc[words.canonical_word_index.astype(int).between(first, last)]
        if len(selection) != last - first + 1:
            raise ValueError("bamboo_canonical_coverage_incomplete")
        start, end = float(selection.iloc[0].start_sec), float(selection.iloc[-1].end_sec)
        if not np.isfinite([start, end]).all() or start >= end:
            raise ValueError("invalid_sentence_boundary")
        spans.append({"sentence_index": sentence_index, "first_word_index": first,
                      "last_word_index": last, "start_sec": start, "end_sec": end})
    return spans


def _reviewed_chunks(audio: np.ndarray, sr: int, frame: pd.DataFrame,
                     start: float, end: float) -> tuple[list[tuple[float, float]], list[np.ndarray]]:
    if not np.isfinite([start, end]).all() or start < 0 or end <= start or end > len(audio) / sr:
        raise ValueError("invalid_timing_interval")
    speech = [(max(start, a), min(end, b)) for a, b in _bounds(frame, "speech")]
    speech = [(a, b) for a, b in speech if b > a]
    spans = _eligible_speech(speech, _bounds(frame, "manual_exclusion"))
    if not spans:
        raise ValueError("no_valid_speech_intervals")
    return spans, [analysis_copy(audio[round(a * sr):round(b * sr)], sr) for a, b in spans]


def _sentence_values(chunks: list[np.ndarray], selected: set[str]) -> tuple[dict, dict]:
    values = {}
    audit = {}
    if any("_wpd_" in name for name in selected):
        wpd, reason, info = wpd_components(chunks)
        audit["wpd"] = info
        for node in range(8):
            entropy_id = f"bamboo_wpd_shannon_entropy_node{node}_norm"
            energy_id = f"bamboo_wpd_energy_share_node{node}"
            if entropy_id in selected:
                value = wpd.get("entropy_norm", np.full(8, np.nan))[node]
                values[entropy_id] = (float(value), reason or (
                    "zero_node_energy" if not np.isfinite(value) else ""))
            if energy_id in selected:
                value = wpd.get("shares", np.full(8, np.nan))[node]
                values[energy_id] = (float(value), reason or (
                    "zero_wavelet_energy" if not np.isfinite(value) else ""))
        if wpd:
            audit["wpd_raw_energy"] = wpd["raw_energy"].tolist()
            audit["wpd_entropy_bits"] = wpd["entropy_bits"].tolist()
    ordinary_needed = any(name.startswith("bamboo_mfcc") for name in selected)
    delta_needed = any(name.startswith("bamboo_dmfcc") for name in selected)
    shared_trajectories = (
        [mfcc_trajectory(chunk[start:start + FS]) for chunk in chunks
         for start in range(0, len(chunk) - FS + 1, FS)]
        if ordinary_needed and delta_needed else None)
    for delta, stem in ((False, "bamboo_mfcc"), (True, "bamboo_dmfcc")):
        if not any(name.startswith(stem) for name in selected):
            continue
        det, reasons, info = rqa_components(chunks, delta=delta,
                                            trajectories=shared_trajectories)
        audit[stem] = info
        for i in range(13):
            name = f"{stem}{i + 1:02d}_rqa_det"
            if name in selected:
                values[name] = (float(det[i]), reasons[i])
    if any(name.startswith(("bamboo_moddepth_", "bamboo_psi_")) for name in selected):
        rhythm, reason, info = rhythm_components(chunks)
        audit["rhythm"] = info
        for name in selected:
            if name.startswith(("bamboo_moddepth_", "bamboo_psi_")):
                values[name] = (float(rhythm.get(name, np.nan)), reason or (
                    "rhythm_component_unavailable" if name not in rhythm else ""))
    visibility_id = "bamboo_regularity_visibility_density_10ms"
    if visibility_id in selected:
        value, reason, info = visibility_10ms(chunks)
        audit["visibility"] = info
        values[visibility_id] = (value, reason)
    return values, audit


def _vowel_values(audio: np.ndarray, sr: int, selected: set[str]) -> tuple[dict, dict]:
    values, audit = {}, {}
    if "vowel_ppe_bits" in selected:
        value, reason, info = ppe_from_audio(audio, sr)
        values["vowel_ppe_bits"] = (value, reason)
        audit["ppe"] = info
    if "vowel_tsfel_shannon_entropy_norm" in selected:
        value, reason, info = tsfel_entropy(analysis_copy(audio, sr))
        values["vowel_tsfel_shannon_entropy_norm"] = (value, reason)
        audit["tsfel"] = info
    if any(name.startswith("vowel_wpd_") for name in selected):
        wpd, reason, info = wpd_components([analysis_copy(audio, sr)])
        audit["wpd"] = info
        for node in range(8):
            name = f"vowel_wpd_energy_share_node{node}"
            if name in selected:
                values[name] = (float(wpd.get("shares", np.full(8, np.nan))[node]), reason)
    if any(name.startswith("vowel_mfcc") for name in selected):
        det, reasons, info = rqa_components([analysis_copy(audio, sr)])
        audit["rqa"] = info
        for i in range(13):
            name = f"vowel_mfcc{i + 1:02d}_rqa_det"
            if name in selected:
                values[name] = (float(det[i]), reasons[i])
    return values, audit


def run_family13_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None,
                       recording_scope=None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions, intervals = final / "final_segmentation_decisions.csv", (
        final / "final_segmentation_intervals.csv")
    if (intervals_csv is None or Path(intervals_csv).resolve() != intervals.resolve()
            or Path(decisions_csv).resolve() != decisions.resolve()):
        raise ValueError("Family 13 requires frozen reviewed segmentation")
    kept = (recording_scope.kept if recording_scope is not None else
            load_final_segmentation(intervals, decisions))
    timeline = (recording_scope.intervals if recording_scope is not None else
                pd.read_csv(intervals, keep_default_na=False))
    project = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    task_id = project_task_key(project)
    selected = set(config.selected_features) & PUBLIC_IDS
    lookup = registry.set_index("feature")
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    tables, native = stage / "tables", stage / "tables" / "native_measurements"
    values_rows, statuses, trial_rows, sentence_rows = [], [], [], []
    review_hash = sha256_file(decisions) + ":" + sha256_file(intervals)
    alignment = None
    alignment_reason = ""
    if task_id == "bamboo_passage" and selected:
        alignment, alignment_reason = load_final_alignment(root)
        if alignment is None:
            alignment_reason = "missing_final_alignment" if alignment_reason == "missing_alignment" else (
                "stale_final_alignment" if "stale" in alignment_reason or
                "mismatch" in alignment_reason else alignment_reason)
    for done, decision in enumerate(kept.itertuples(), 1):
        recording_id = str(decision.recording_id)
        frame = timeline.loc[timeline.recording_id.astype(str).eq(recording_id)].copy()
        if "view" in frame:
            frame = frame.loc[frame["view"].astype(str).eq("authoritative")]
        base = {"recording_id": recording_id, "task_id": task_id,
                "source_audio_path": str(getattr(decision, "source_file_path", "")),
                "source_audio_sha256": str(getattr(decision, "source_sha256", "")),
                "segmentation_review_hash": review_hash,
                "input_contract": ("FINAL_ALIGNMENT_CONTRACT_V1" if task_id == "bamboo_passage"
                                   else "FINAL_REVIEWED_SEGMENTATION"),
                "alignment_run_id": (alignment.manifest.get("alignment_run_id", "")
                                     if alignment is not None else ""),
                "algorithm_version": VERSION}
        trials = []
        issue = ""
        if task_id == "sustained_a":
            speech = _bounds(frame, "speech")
            if speech:
                trials = [{"trial_id": f"{recording_id}:reviewed_001",
                           "sequence_id": f"{recording_id}:reviewed_001",
                           "start_sec": min(a for a, _ in speech),
                           "end_sec": max(b for _, b in speech)}]
        elif task_id == "wstg":
            trials, issue = load_confirmed_trials(root, recording_id, task_id)
        elif task_id == "bamboo_passage" and alignment is not None:
            trials = alignment.trials.loc[
                alignment.trials.recording_id.astype(str).eq(recording_id)
            ].to_dict("records")
        elif task_id == "bamboo_passage":
            issue = alignment_reason
        else:
            issue = "task_not_applicable"
        try:
            source = Path(base["source_audio_path"])
            if not source.is_file():
                raise ValueError("native_source_audio_missing")
            audio, sr = sf.read(source, dtype="float64", always_2d=False)
            if audio.ndim != 1:
                raise ValueError("native_audio_not_mono")
        except (OSError, ValueError) as exc:
            audio, sr = None, 0
            issue = str(exc)
        for trial in trials:
            trial_id = str(trial["trial_id"])
            sequence_id = str(trial["sequence_id"])
            trial_values, trial_audit = {}, {}
            trial_issue = ""
            try:
                if audio is None:
                    raise ValueError(issue)
                if task_id == "sustained_a":
                    spans = centered_stable_spans(decision, frame, len(audio) / sr, [trial])
                    start, end, *_ = next(span for span in spans if span[2] == trial_id)
                    signal = audio[round(start * sr):round(end * sr)].copy()
                    trial_values, trial_audit = _vowel_values(signal, int(sr), selected)
                elif task_id == "wstg":
                    spans, chunks = _reviewed_chunks(
                        audio, int(sr), frame, float(trial["start_sec"]), float(trial["end_sec"]))
                    if "s1_tsfel_shannon_entropy_norm" in selected:
                        measurements = [tsfel_entropy(chunk) for chunk in chunks]
                        valid = [value for value, reason, _ in measurements if not reason]
                        reason = "" if valid else measurements[0][1]
                        trial_values["s1_tsfel_shannon_entropy_norm"] = (
                            float(np.median(valid)) if valid else np.nan, reason)
                        trial_audit["tsfel"] = [row[2] for row in measurements]
                else:
                    sentence_spans = _sentence_spans(alignment, recording_id, trial_id)
                    for sentence in sentence_spans:
                        try:
                            spans, chunks = _reviewed_chunks(
                                audio, int(sr), frame, sentence["start_sec"], sentence["end_sec"])
                            values, audit = _sentence_values(chunks, selected)
                            reason = ""
                        except ValueError as exc:
                            spans, values, audit, reason = [], {}, {}, str(exc)
                        sentence_rows.append({**base, "trial_id": trial_id,
                                              "sequence_id": sequence_id, **sentence,
                                              "eligible_intervals": json.dumps(spans),
                                              "failure_reason": reason,
                                              "audit_json": json.dumps(audit, sort_keys=True),
                                              **{key: value for key, (value, _) in values.items()},
                                              **{key + "_reason": why for key, (_, why) in values.items()}})
                    own = [row for row in sentence_rows if row["recording_id"] == recording_id
                           and row["trial_id"] == trial_id]
                    for feature in selected:
                        valid = [float(row[feature]) for row in own if feature in row
                                 and not row.get(feature + "_reason")
                                 and np.isfinite(row[feature])]
                        reason = "" if len(valid) >= MIN_SENTENCES else (
                            "insufficient_valid_sentences" if valid else
                            next((row.get(feature + "_reason") or row.get("failure_reason")
                                  for row in own if row.get(feature + "_reason") or
                                  row.get("failure_reason")), "insufficient_valid_sentences"))
                        trial_values[feature] = (
                            float(np.median(valid)) if len(valid) >= MIN_SENTENCES else np.nan,
                            reason)
                    trial_audit["n_sentences"] = len(sentence_spans)
            except (ValueError, StopIteration) as exc:
                trial_issue = str(exc)
            trial_rows.append({**base, "trial_id": trial_id, "sequence_id": sequence_id,
                               "trial_start_sec": trial["start_sec"], "trial_end_sec": trial["end_sec"],
                               "native_sample_rate_hz": sr, "analysis_sample_rate_hz": FS,
                               "resampler": "scipy.signal.resample_poly",
                               "resampler_version": SCIPY_VERSION,
                               "failure_reason": trial_issue,
                               "audit_json": json.dumps(trial_audit, sort_keys=True),
                               **{key: value for key, (value, _) in trial_values.items()},
                               **{key + "_reason": why for key, (_, why) in trial_values.items()}})
        own = [row for row in trial_rows if row["recording_id"] == recording_id]
        output = dict(base)
        for feature in sorted(selected):
            target_task = next(row["task_id"] for row in specs() if row["feature_id"] == feature)
            valid = [float(row[feature]) for row in own if feature in row
                     and not row.get(feature + "_reason") and np.isfinite(row[feature])]
            reason = "" if valid else next(
                (row.get(feature + "_reason") or row.get("failure_reason")
                 for row in own if row.get(feature + "_reason") or row.get("failure_reason")),
                issue or "missing_reviewed_segmentation")
            if target_task != task_id:
                reason = "task_not_applicable"
            value = float(np.median(valid)) if valid and not reason else np.nan
            output[feature] = value
            statuses.append({**base, "feature": feature, "feature_id": feature,
                             "family_id": "F13", "value": value,
                             "unit": str(lookup.loc[feature, "unit"]),
                             "status": "computed" if not reason else "unavailable",
                             "failure_reason": reason, "valid_trial_count": len(valid),
                             "trial_count": len(trials)})
        values_rows.append(output)
        if progress_callback:
            progress_callback(done, len(kept), f"Acoustic Features — {decision.file_name}")
    native.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    trial_path = native / "family13_trial_values.csv"
    sentence_path = native / "family13_sentence_values.csv"
    value_frame, status_frame = pd.DataFrame(values_rows), pd.DataFrame(statuses)
    value_frame.to_csv(values_path, index=False)
    status_frame.to_csv(status_path, index=False)
    registry.to_csv(registry_path, index=False)
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    pd.DataFrame(sentence_rows).to_csv(sentence_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, value_frame, registry, status_frame,
                                        "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff,
                               optional_main={trial_path.name: trial_path,
                                              sentence_path.name: sentence_path})
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version=VERSION,
        status="completed_with_warnings" if status_frame.failure_reason.astype(str).ne("").any()
        else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          ArtifactRef(str(trial_path), "family13_trial_values", "text/csv"),
                          ArtifactRef(str(sentence_path), "family13_sentence_values", "text/csv")],
        config={"selected_features": sorted(selected), "algorithm_version": VERSION,
                "sentence_minimum": MIN_SENTENCES})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
