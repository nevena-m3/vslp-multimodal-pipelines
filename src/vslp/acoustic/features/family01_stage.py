"""Reviewed native-rate Family 01 extraction and track audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vslp.acoustic.alignment.trials import confirmed_task_trials
from vslp.acoustic.features.family01 import (
    FAMILY01_ALIASES, FAMILY01_IDS, F0Track, MIN_SUSTAINED_YIELD, MIN_VALID_FRAMES,
    PITCH_CEILING_HZ, PITCH_FLOOR_HZ, TIME_STEP_SEC,
    calculate_f0_features, praat_f0_track,
)
from vslp.acoustic.features.family01_task import task_context
from vslp.acoustic.features.family02 import FAMILY02_IDS, calculate_voice_quality
from vslp.acoustic.segment.review import load_final_segmentation
from vslp.core.feature_contract import build_feature_delivery, write_feature_handoff
from vslp.core.provenance import sha256_file
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult


def _trial_at_time(trials: list[dict], time_sec: float) -> str:
    matches = [str(item["trial_id"]) for item in trials
               if float(item["start_sec"]) <= time_sec < float(item["end_sec"])]
    return matches[0] if len(matches) == 1 else ""


def _region_tracks(row, intervals: pd.DataFrame, audio: np.ndarray, sr: int,
                   family: str, trials: list[dict]):
    """Track each original-time eligible span; never concatenate audio across gaps."""
    sustained = family == "sustained_vowel"
    speech = intervals.loc[intervals.segment_role.eq("speech")].sort_values("start_sec")
    exclusions = [(float(item.start_sec), float(item.end_sec)) for item in
                  intervals.loc[intervals.segment_role.eq("manual_exclusion")].itertuples()]
    if sustained:
        start, end = float(row.stable_region_start), float(row.stable_region_end)
        domain = (start, end) if 0 <= start < end <= len(audio) / sr else None
        if domain is None and not any("stable_region_start" in trial and
                                      "stable_region_end" in trial for trial in trials):
            raise ValueError("stable_phonation_region_unavailable")
        region = "final_reviewed_stable_phonation"
    else:
        domain = (0.0, len(audio) / sr)
        region = {"bamboo_passage": "reviewed_passage_voiced_frames",
                  "sentence": "reviewed_sentence_voiced_frames",
                  "ddk": "reviewed_ddk_voiced_frames"}.get(family, "reviewed_task_voiced_frames")
    spans: list[tuple[float, float, str, float, float]] = []
    for item in speech.itertuples():
        left, right = float(item.start_sec), float(item.end_sec)
        if not sustained:
            left, right = max(left, domain[0]), min(right, domain[1])
        if left >= right:
            continue
        trial_domains = trials or [{"trial_id": "", "start_sec": left, "end_sec": right}]
        for trial in trial_domains:
            trial_id = str(trial["trial_id"])
            trial_start, trial_end = float(trial["start_sec"]), float(trial["end_sec"])
            stable = (float(trial["stable_region_start"]),
                      float(trial["stable_region_end"])) if (
                          sustained and "stable_region_start" in trial and
                          "stable_region_end" in trial) else domain
            if sustained and stable is None:
                continue
            start, end = max(left, trial_start), min(right, trial_end)
            if sustained:
                start, end = max(start, stable[0]), min(end, stable[1])
            fragments = [(start, end)] if start < end else []
            for excluded_start, excluded_end in exclusions:
                fragments = ([(a, min(b, excluded_start)) for a, b in fragments
                              if a < min(b, excluded_start)] +
                             [(max(a, excluded_end), b) for a, b in fragments
                              if max(a, excluded_end) < b])
            spans.extend((a, b, trial_id, trial_start, trial_end) for a, b in fragments)
    if sustained:
        # A central phonation window must be contiguous: never bridge an exclusion.
        by_trial: dict[str, list[tuple[float, float, str, float, float]]] = {}
        for span in spans:
            by_trial.setdefault(span[2], []).append(span)
        centered = []
        for candidates in by_trial.values():
            start, end, trial_id, trial_start, trial_end = max(
                candidates, key=lambda span: (span[1] - span[0], -span[0]))
            width = min(2.0, end - start)
            middle = (start + end) / 2.0
            centered.append((middle - width / 2.0, middle + width / 2.0,
                             trial_id, trial_start, trial_end))
        spans = centered
    if not spans:
        raise ValueError("final_reviewed_speech_region_unavailable")
    span_tracks = []
    for start, end, trial_id, trial_start, trial_end in sorted(spans):
        segment = audio[round(start * sr):round(end * sr)]
        if segment.size < int(sr * .1):
            continue
        native = praat_f0_track(segment, sr, analysis_start_sec=start,
                                analysis_region=region)
        span_tracks.append((native, trial_id, trial_start, trial_end))
    if not span_tracks:
        raise ValueError("analysis_region_too_short")
    tracks = [item[0] for item in span_tracks]
    track = F0Track(np.concatenate([t.time_sec for t in tracks]),
                    np.concatenate([t.f0_hz for t in tracks]),
                    np.concatenate([t.voiced_valid for t in tracks]), sr,
                    min(t.analysis_start_sec for t in tracks),
                    max(t.analysis_end_sec for t in tracks), region)
    return track, sustained, span_tracks


def _f0_qc(span_tracks: list[tuple]) -> dict[str, bool]:
    """Technical review flags only; no F0 values are repaired or discarded."""
    halving = doubling = floor = ceiling = False
    for track, *_ in span_tracks:
        hz, valid = track.f0_hz, track.voiced_valid
        floor |= bool(np.any(valid & (hz <= PITCH_FLOOR_HZ)))
        ceiling |= bool(np.any(valid & (hz >= PITCH_CEILING_HZ)))
        adjacent = valid[1:] & valid[:-1]
        ratios = hz[1:][adjacent] / hz[:-1][adjacent]
        halving |= bool(np.any((ratios >= .45) & (ratios <= .55)))
        doubling |= bool(np.any((ratios >= 1.8) & (ratios <= 2.2)))
    return {"possible_pitch_halving": halving, "possible_pitch_doubling": doubling,
            "possible_octave_jump": halving or doubling,
            "floor_contact": floor, "ceiling_contact": ceiling}


def _f01_interpretation(feature_id: str, family: str) -> str:
    if feature_id in {"pfr_maxmin_st", "observed_f0_range_p05_p95_st"}:
        return "Observed F0 span within the analyzed task; not elicited physiologic range"
    if feature_id in FAMILY01_ALIASES:
        return "Deprecated alias of sentence-level primitive F0 variability; not a latent factor"
    if feature_id in {"f0_sd_st", "f0_iqr_st", "f0_range_st"}:
        return ("Within-phonation F0 variability" if family == "sustained_vowel" else
                "Task-specific connected-speech/prosodic F0 variability")
    return "Task-specific valid voiced-frame F0 level"


def _approved_sentence_intonation(track: F0Track, intervals: pd.DataFrame,
                                  audit_rows: list[dict] | None = None) -> dict[str, float] | None:
    """Use explicit approved sentence labels in final reviewed segmentation, if present."""
    required = {"sentence_id", "sentence_review_status", "segment_role",
                "start_sec", "end_sec"}
    if not required <= set(intervals.columns):
        return None
    sentences = intervals.loc[(intervals.segment_role == "speech") &
                              (intervals.sentence_id.astype(str).str.len() > 0)]
    if sentences.empty or not sentences.sentence_review_status.astype(str).eq("APPROVED").all():
        return None
    counts = np.zeros(track.time_sec.size, dtype=int)
    values: dict[str, list[float]] = {key: [] for key in ("f0_sd_st", "f0_iqr_st")}
    pending_rows = []
    for sentence_index, (sentence_id, group) in enumerate(
            sentences.groupby("sentence_id", sort=False), 1):
        mask = np.zeros(track.time_sec.size, dtype=bool)
        for item in group.itertuples():
            mask |= (track.time_sec >= float(item.start_sec)) & (
                track.time_sec < float(item.end_sec))
        counts += mask.astype(int)
        if not mask.any():
            continue
        sentence_track = F0Track(track.time_sec[mask], track.f0_hz[mask],
                                 track.voiced_valid[mask], track.sample_rate_hz,
                                 float(group.start_sec.min()),
                                 float(group.end_sec.max()), "approved_reviewed_sentence")
        calculated = calculate_f0_features(sentence_track, sustained=False)
        pending_rows.append({
            "sentence_id": str(sentence_id), "sentence_index": sentence_index,
            "analysis_start_sec": sentence_track.analysis_start_sec,
            "analysis_end_sec": sentence_track.analysis_end_sec,
            "analysis_duration_sec": (sentence_track.analysis_end_sec -
                                      sentence_track.analysis_start_sec),
            "n_candidate_frames": int(sentence_track.f0_hz.size),
            "n_valid_f0_frames": int(sentence_track.voiced_valid.sum()),
            "tracking_yield": sentence_track.tracking_yield,
            "f0_sd_st": calculated["f0_sd_st"][0],
            "f0_iqr_st": calculated["f0_iqr_st"][0],
            "qc_status": calculated["f0_sd_st"][1] or "valid",
            "f0_qc_flags": json.dumps(_f0_qc([(sentence_track,)]), sort_keys=True),
            "construct_role": "intonation_component",
            "intonation_analysis_unit": "sentence",
        })
        for key in values:
            if not calculated[key][1]:
                values[key].append(calculated[key][0])
    if not np.all(counts[track.voiced_valid] == 1):
        return None
    if audit_rows is not None:
        audit_rows.extend(pending_rows)
    return {key: float(np.median(numbers)) for key, numbers in values.items()
            if numbers} or None


def run_family01_stage(decisions_csv: str | Path, root: str | Path, config,
                       intervals_csv: str | Path | None, registry: pd.DataFrame,
                       progress_callback=None, execution_stage_dir: Path | None = None) -> StageResult:
    root = Path(root)
    final = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = final / "final_segmentation_decisions.csv"
    intervals = final / "final_segmentation_intervals.csv"
    if (Path(decisions_csv).resolve() != decisions.resolve() or intervals_csv is None
            or Path(intervals_csv).resolve() != intervals.resolve()):
        raise ValueError("Family 01 requires authoritative frozen reviewed segmentation")
    kept = load_final_segmentation(intervals, decisions)
    all_intervals = pd.read_csv(intervals, keep_default_na=False)
    run = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    selected = set(config.selected_features) & (FAMILY01_IDS | FAMILY02_IDS)
    selected01 = selected & FAMILY01_IDS
    selected_primary = {FAMILY01_ALIASES.get(feature_id, feature_id)
                        for feature_id in selected01}
    selected02 = selected & FAMILY02_IDS
    stage = execution_stage_dir or root / "acoustic" / "005_features"
    values, statuses, track_rows, pulse_rows, trial_rows = [], [], [], [], []
    legacy_rows = []
    sentence_rows = []
    task = task_context(run)
    f01_profile = json.loads((Path(__file__).with_name("data") / "family01.json").read_text(
        encoding="utf-8"))
    if progress_callback:
        progress_callback(0, len(kept), "Acoustic Features")
    for index, row in enumerate(kept.itertuples(), 1):
        record_id = str(row.recording_id)
        confirmed_trials = confirmed_task_trials(root, record_id)
        timeline = all_intervals.loc[all_intervals.recording_id.astype(str).eq(record_id)]
        result = {feature_id: (np.nan, "analysis_unavailable") for feature_id in selected}
        track = None
        span_tracks = []
        qc_flags: dict[str, bool] = {}
        intonation_analysis_unit = (
            "sentence" if task["task_family"] == "sentence" else "not_applicable")
        used_region = "unavailable"
        used_start = np.nan
        used_end = np.nan
        used_duration = np.nan
        cycle_audit = {"n_cycles": 0, "n_pulses": 0, "hnr_valid_frames": 0}
        try:
            audio, sr = sf.read(row.analysis_wav_path, dtype="float32", always_2d=False)
            if audio.ndim != 1:
                raise ValueError("canonical_audio_not_mono")
            if selected01:
                try:
                    track, sustained, span_tracks = _region_tracks(
                        row, timeline, audio, int(sr), task["task_family"], confirmed_trials)
                    used_region, used_start, used_end = (track.analysis_region,
                                                         track.analysis_start_sec,
                                                         track.analysis_end_sec)
                    used_duration = sum(native.analysis_end_sec - native.analysis_start_sec
                                        for native, *_ in span_tracks)
                    qc_flags = _f0_qc(span_tracks)
                    qc_flags["insufficient_f0_frames"] = bool(
                        track.voiced_valid.sum() < MIN_VALID_FRAMES)
                    qc_flags["low_tracking_yield"] = bool(
                        sustained and track.tracking_yield < MIN_SUSTAINED_YIELD)
                    pooled = calculate_f0_features(track, sustained=sustained)
                    legacy_rows.append({"recording_id": record_id,
                                        "aggregation_policy": "legacy_pooled_valid_frames_v1",
                                        **{key: pooled[key][0] for key in selected_primary}})
                    for span_index, (native, trial_id, trial_start, trial_end) in enumerate(
                            span_tracks, 1):
                        track_rows.extend({
                            "recording_id": record_id, "file_name": row.file_name,
                            "task_family": task["task_family"], "task_item": task["task_item"],
                            "time_sec": float(t), "f0_hz": float(hz), "voiced_valid": bool(valid),
                            "trial_id": trial_id, "trial_start_sec": trial_start,
                            "trial_end_sec": trial_end, "analysis_span_index": span_index,
                            "analysis_start_sec": native.analysis_start_sec,
                            "analysis_end_sec": native.analysis_end_sec,
                            "analysis_duration_sec": native.analysis_end_sec - native.analysis_start_sec,
                            "analysis_region": native.analysis_region,
                            "algorithm_version": native.algorithm_version,
                            "parameter_set_id": native.parameter_set_id, "sample_rate_hz": int(sr),
                        } for t, hz, valid in zip(native.time_sec, native.f0_hz,
                                                   native.voiced_valid, strict=True))
                    trial_groups = confirmed_trials or [{"trial_id": "", "start_sec": used_start,
                                                          "end_sec": used_end}]
                    intonation_modes: set[str] = set()
                    trial_results: list[dict[str, tuple[float, str]]] = []
                    for trial in trial_groups:
                        trial_id = str(trial["trial_id"])
                        parts = [native for native, identity, *_ in span_tracks
                                 if identity == trial_id]
                        if parts:
                            trial_track = F0Track(
                                np.concatenate([part.time_sec for part in parts]),
                                np.concatenate([part.f0_hz for part in parts]),
                                np.concatenate([part.voiced_valid for part in parts]), int(sr),
                                min(part.analysis_start_sec for part in parts),
                                max(part.analysis_end_sec for part in parts), used_region)
                            trial_values = calculate_f0_features(trial_track, sustained=sustained)
                            trial_qc = _f0_qc([(part,) for part in parts])
                            trial_qc["insufficient_f0_frames"] = bool(
                                trial_track.voiced_valid.sum() < MIN_VALID_FRAMES)
                            trial_qc["low_tracking_yield"] = bool(
                                sustained and trial_track.tracking_yield < MIN_SUSTAINED_YIELD)
                            sentence_values = None
                            if task["task_family"] == "bamboo_passage":
                                pending_sentences: list[dict] = []
                                sentence_values = _approved_sentence_intonation(
                                    trial_track, timeline, pending_sentences)
                                intonation_analysis_unit = (
                                    "sentence" if sentence_values else "passage_fallback")
                                intonation_modes.add(intonation_analysis_unit)
                                if sentence_values:
                                    sentence_rows.extend({"recording_id": record_id,
                                                          "task_family": task["task_family"],
                                                          "task_item": task["task_item"],
                                                          "trial_id": trial_id,
                                                          "pitch_floor_hz": PITCH_FLOOR_HZ,
                                                          "pitch_ceiling_hz": PITCH_CEILING_HZ,
                                                          "time_step_sec": TIME_STEP_SEC,
                                                          "algorithm_version": track.algorithm_version,
                                                          "parameter_set_id": track.parameter_set_id,
                                                          **item} for item in pending_sentences)
                                else:
                                    sentence_rows.append({
                                        "recording_id": record_id, "trial_id": trial_id,
                                        "task_family": task["task_family"],
                                        "task_item": task["task_item"],
                                        "pitch_floor_hz": PITCH_FLOOR_HZ,
                                        "pitch_ceiling_hz": PITCH_CEILING_HZ,
                                        "time_step_sec": TIME_STEP_SEC,
                                        "algorithm_version": track.algorithm_version,
                                        "parameter_set_id": track.parameter_set_id,
                                        "sentence_id": "", "sentence_index": np.nan,
                                        "analysis_start_sec": trial_track.analysis_start_sec,
                                        "analysis_end_sec": trial_track.analysis_end_sec,
                                        "analysis_duration_sec": (trial_track.analysis_end_sec -
                                                                  trial_track.analysis_start_sec),
                                        "n_candidate_frames": int(trial_track.f0_hz.size),
                                        "n_valid_f0_frames": int(trial_track.voiced_valid.sum()),
                                        "tracking_yield": trial_track.tracking_yield,
                                        "f0_sd_st": trial_values["f0_sd_st"][0],
                                        "f0_iqr_st": trial_values["f0_iqr_st"][0],
                                        "qc_status": trial_values["f0_sd_st"][1] or "valid",
                                        "f0_qc_flags": json.dumps(trial_qc, sort_keys=True),
                                        "construct_role": "intonation_component",
                                        "intonation_analysis_unit": "passage_fallback",
                                    })
                        else:
                            trial_track = None
                            trial_values = {feature_id: (np.nan, "no_eligible_trial_frames")
                                            for feature_id in selected_primary}
                            trial_qc = {}
                            sentence_values = None
                        output_values = dict(trial_values)
                        for alias, primitive in FAMILY01_ALIASES.items():
                            if alias in selected01:
                                output_values[alias] = (
                                    (sentence_values[primitive], "") if sentence_values and
                                    primitive in sentence_values else trial_values[primitive])
                        trial_results.append(output_values)
                        trial_rows.append({
                            "recording_id": record_id, "file_name": row.file_name,
                            "task_family": task["task_family"], "task_item": task["task_item"],
                            "trial_id": trial_id, "trial_start_sec": float(trial["start_sec"]),
                            "trial_end_sec": float(trial["end_sec"]),
                            "analysis_start_sec": (trial_track.analysis_start_sec
                                                   if trial_track else np.nan),
                            "analysis_end_sec": (trial_track.analysis_end_sec
                                                 if trial_track else np.nan),
                            "analysis_duration_sec": sum(part.analysis_end_sec -
                                                         part.analysis_start_sec for part in parts),
                            "n_candidate_frames": int(trial_track.f0_hz.size) if trial_track else 0,
                            "n_valid_f0_frames": int(trial_track.voiced_valid.sum())
                            if trial_track else 0,
                            "tracking_yield": trial_track.tracking_yield if trial_track else 0.0,
                            "f0_qc_flags": json.dumps(trial_qc, sort_keys=True),
                            "feature_failures": json.dumps({feature_id: output_values[feature_id][1]
                                                            for feature_id in selected01
                                                            if output_values[feature_id][1]},
                                                           sort_keys=True),
                            "aggregation_policy": "valid_frames_within_trial",
                            "intonation_analysis_unit": intonation_analysis_unit,
                            "algorithm_version": track.algorithm_version,
                            "parameter_set_id": track.parameter_set_id,
                            **{feature_id: trial_values[feature_id][0]
                               for feature_id in selected_primary},
                        })
                    for feature_id in selected01:
                        valid_trials = [float(item[feature_id][0]) for item in trial_results
                                        if not item[feature_id][1] and
                                        np.isfinite(item[feature_id][0])]
                        result[feature_id] = (
                            (float(np.median(valid_trials)), "") if valid_trials else
                            (np.nan, "no_valid_trial_measurement"))
                    if len(intonation_modes) > 1:
                        intonation_analysis_unit = "mixed_units_unavailable"
                        for feature_id in selected01 & {
                                "intonation_f0_sd_st", "intonation_f0_iqr_st"}:
                            result[feature_id] = (np.nan, "mixed_sentence_and_passage_units")
                except (ValueError, RuntimeError) as exc:
                    result.update({k: (np.nan, str(exc).splitlines()[0]) for k in selected01})
            if selected02:
                try:
                    if str(row.segmentation_method) != "sustained_phonation":
                        raise ValueError("stable_phonation_region_unavailable")
                    start, end = float(row.stable_region_start), float(row.stable_region_end)
                    if not 0 <= start < end <= len(audio) / sr:
                        raise ValueError("stable_phonation_region_unavailable")
                    region_audio = audio[round(start * sr):round(end * sr)]
                    used_region, used_start, used_end = (
                        "final_reviewed_stable_phonation", start, end)
                    voice_values, cycle_audit = calculate_voice_quality(region_audio, int(sr))
                    result.update({k: v for k, v in voice_values.items() if k in selected02})
                    times = cycle_audit.get("pulse_time_sec", np.array([]))
                    periods = cycle_audit.get("period_sec", np.array([]))
                    pulse_rows.extend({"recording_id": record_id, "file_name": row.file_name,
                                       "pulse_index": i + 1, "time_sec": float(start + t),
                                       "trial_id": _trial_at_time(confirmed_trials, float(start + t)),
                                       "following_period_sec": float(periods[i]) if i < len(periods) else np.nan,
                                       "analysis_region": "final_reviewed_stable_phonation",
                                       "algorithm_version": "family02-praat-1.0.0",
                                       "parameter_set_id": "family02_stable_vowel_praat_v1"}
                                      for i, t in enumerate(times))
                except (ValueError, RuntimeError) as exc:
                    result.update({k: (np.nan, str(exc)) for k in selected02})
        except (OSError, ValueError, RuntimeError) as exc:
            reason = str(exc).splitlines()[0]
            result = {feature_id: (np.nan, reason) for feature_id in selected}
        base = {
            "recording_id": record_id, "file_name": row.file_name,
            "task_id": run.get("task_id", ""), "task_type": run.get("task_type", ""),
            "task_family": task["task_family"], "task_item": task["task_item"],
            "stimulus_text": task["stimulus_text"],
            "task_name": run["task_name"], "run_id": run["run_id"],
            "source_sha256": row.source_sha256, "source_file_path": row.source_file_path,
            "analysis_wav_path": row.analysis_wav_path,
            "segmentation_run_id": row.segmentation_run_id, "review_run_id": row.review_run_id,
            "analysis_region": used_region,
            "analysis_unit": "recording",
            "analysis_start_sec": used_start,
            "analysis_end_sec": used_end,
            "analysis_duration_sec": used_duration,
            "n_candidate_frames": int(track.f0_hz.size) if track else 0,
            "n_valid_f0_frames": int(track.voiced_valid.sum()) if track else 0,
            "n_valid_frames": int(track.voiced_valid.sum()) if track else 0,
            "tracking_yield": track.tracking_yield if track else np.nan,
            "f0_qc_flags": json.dumps(qc_flags, sort_keys=True),
            "f0_qc_flag_policy_id": f01_profile["qc_flag_policy_id"],
            "pitch_floor_hz": PITCH_FLOOR_HZ, "pitch_ceiling_hz": PITCH_CEILING_HZ,
            "time_step_sec": TIME_STEP_SEC,
            "stable_region_policy_id": f01_profile["stable_region_policy_id"],
            "recording_aggregation_policy_id": f01_profile["recording_aggregation_policy_id"],
            "intonation_analysis_unit": intonation_analysis_unit,
            "validation_status": json.dumps(f01_profile["validation_status"], sort_keys=True),
            "n_cycles": cycle_audit["n_cycles"],
            "n_pulses": cycle_audit["n_pulses"],
            "hnr_valid_frames": cycle_audit["hnr_valid_frames"],
            "final_decisions_sha256": sha256_file(decisions),
            "final_intervals_sha256": sha256_file(intervals),
        }
        values.append({**base, **{feature_id: result[feature_id][0] for feature_id in selected}})
        for feature_id in sorted(selected):
            value, reason = result[feature_id]
            spec = registry.loc[registry.feature.eq(feature_id)].iloc[0]
            statuses.append({**base, "feature": feature_id, "feature_id": feature_id,
                             "family_id": spec.family_id, "value": value, "unit": spec.unit,
                             "analysis_unit": ("sentence" if feature_id in FAMILY01_ALIASES and
                                               intonation_analysis_unit == "sentence" else
                                               "passage_fallback" if feature_id in FAMILY01_ALIASES and
                                               intonation_analysis_unit == "passage_fallback" else
                                               "recording median of valid trial values"
                                               if feature_id in selected01 else "stable phonation"),
                             "construct_role": ("intonation_component" if feature_id in
                                                FAMILY01_ALIASES or feature_id in {"f0_sd_st",
                                                "f0_iqr_st"} and intonation_analysis_unit ==
                                                "sentence" else "f0_variability" if feature_id in
                                                {"f0_sd_st", "f0_iqr_st"} else ""),
                             "deprecated_alias_of": FAMILY01_ALIASES.get(feature_id, ""),
                             "aggregation_method": ("median_valid_trials_v1"
                                                    if feature_id in selected01 else ""),
                             "scientific_interpretation": _f01_interpretation(
                                 feature_id, task["task_family"]) if feature_id in selected01 else "",
                             "native_measurement_table": "f0_tracks.csv" if feature_id in selected01 else "",
                             "trial_measurement_table": (
                                 "f0_sentence_summaries.csv" if feature_id in
                                 FAMILY01_ALIASES and task["task_family"] == "bamboo_passage"
                                 else "f0_trial_summaries.csv" if feature_id in selected01 else ""),
                             "algorithm": spec.algorithm_version, "algorithm_version": spec.algorithm_version,
                             "parameter_set_id": spec.parameter_set_id,
                             "status": "unavailable" if reason else "computed",
                             "failure_reason": reason})
        if progress_callback:
            progress_callback(index, len(kept), f"Acoustic Features — {row.file_name}")
    tables = stage / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    values_path = tables / "acoustic_features_per_file.csv"
    status_path = tables / "acoustic_feature_status_long.csv"
    registry_path = tables / "selected_acoustic_feature_registry.csv"
    track_path = tables / "native_measurements" / "f0_tracks.csv" if selected01 else None
    trial_path = tables / "native_measurements" / "f0_trial_summaries.csv" if selected01 else None
    sentence_path = (tables / "native_measurements" / "f0_sentence_summaries.csv"
                     if selected01 and task["task_family"] == "bamboo_passage" else None)
    legacy_path = tables / "native_measurements" / "f0_legacy_pooled_recording.csv" if selected01 else None
    pulse_path = tables / "native_measurements" / "voice_pulses.csv" if selected02 else None
    pd.DataFrame(values).to_csv(values_path, index=False)
    pd.DataFrame(statuses).to_csv(status_path, index=False)
    if track_path:
        track_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(track_rows).to_csv(track_path, index=False)
        pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
        pd.DataFrame(legacy_rows).to_csv(legacy_path, index=False)
        if sentence_path:
            pd.DataFrame(sentence_rows).to_csv(sentence_path, index=False)
    if pulse_path:
        pulse_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pulse_rows).to_csv(pulse_path, index=False)
    registry.to_csv(registry_path, index=False)
    if execution_stage_dir is None:
        handoff = write_feature_handoff(tables, pd.DataFrame(values), registry,
                                        pd.DataFrame(statuses), "acoustic", values_path, registry_path)
        build_feature_delivery(root, "acoustic", handoff)
    manifest = StageManifest(
        stage_name="acoustic_features", stage_version="reviewed-voice-1.0.0",
        status="completed_with_warnings" if any(s["failure_reason"] for s in statuses) else "completed",
        input_artifacts=[ArtifactRef(str(decisions), "final_segmentation_decisions", "text/csv"),
                         ArtifactRef(str(intervals), "final_segmentation_intervals", "text/csv")],
        output_artifacts=[ArtifactRef(str(values_path), "feature_values", "text/csv"),
                          ArtifactRef(str(status_path), "feature_status", "text/csv"),
                          *([ArtifactRef(str(track_path), "shared_f0_track", "text/csv")] if track_path else []),
                          *([ArtifactRef(str(trial_path), "f0_trial_summaries", "text/csv")]
                            if trial_path else []),
                          *([ArtifactRef(str(sentence_path), "f0_sentence_summaries", "text/csv")]
                            if sentence_path else []),
                          *([ArtifactRef(str(legacy_path), "f0_legacy_pooled_recording", "text/csv")]
                            if legacy_path else []),
                          *([ArtifactRef(str(pulse_path), "praat_pulses", "text/csv")] if pulse_path else [])],
        config={"selected_features": sorted(selected),
                "parameter_set_ids": sorted({s["parameter_set_id"] for s in statuses}),
                "task_family": task["task_family"], "task_item": task["task_item"],
                "stable_region_policy_id": f01_profile["stable_region_policy_id"],
                "f0_qc_flag_policy_id": f01_profile["qc_flag_policy_id"],
                "f0_aggregation_policy": "median of valid trial-level values; legacy pooled recording values retained separately",
                "recording_aggregation_policy_id": f01_profile["recording_aggregation_policy_id"],
                "f0_qc_flags_are_technical_only": True,
                "source_audio_policy": "canonical native rate; read only"})
    manifest_path = stage / "logs" / "stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return StageResult(manifest.status, manifest_path, values_path)
