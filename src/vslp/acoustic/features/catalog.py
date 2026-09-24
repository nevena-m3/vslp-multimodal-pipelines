"""Source-bound registry for the 79 master-matrix constructs.

The matrix has no exact output IDs. Family specifications may add leaf outputs;
until then no feature algorithm is approved for production execution.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_DATA = Path(__file__).with_name("data") / "master_matrix.json"
_FAMILY01 = Path(__file__).with_name("data") / "family01.json"
_FAMILY02 = Path(__file__).with_name("data") / "family02.json"
_FAMILY04 = Path(__file__).with_name("data") / "family04.json"
_FAMILY05 = Path(__file__).with_name("data") / "family05.json"
_FAMILY06 = Path(__file__).with_name("data") / "family06.json"
_FAMILY07 = Path(__file__).with_name("data") / "family07.json"
_FAMILY08 = Path(__file__).with_name("data") / "family08.json"
_FAMILY09 = Path(__file__).with_name("data") / "family09.json"
_FAMILY10 = Path(__file__).with_name("data") / "family10.json"
_FAMILY11 = Path(__file__).with_name("data") / "family11.json"
_FAMILY12 = Path(__file__).with_name("data") / "family12.json"
_FAMILY13 = Path(__file__).with_name("data") / "family13.json"
USE = {"PROD": "Production", "VARIANT": "Conditional", "TASK": "Conditional",
       "VALIDATE": "Validation required", "RESEARCH": "Research", "BLOCKED": "Blocked"}
TASKS = {
    "sustained_a": "Sustained /a/",
    "bamboo_passage": "Bamboo Passage",
    "ddk": "DDK",
    "wstg": "WSTG",
}
REGISTERED_EXECUTORS = frozenset({"F01", "F02", "F04", "F06", "F07", "F08", "F09",
                                  "F10", "F11", "F12"})


def validate_feature_ids(outputs: list[dict[str, Any]]) -> None:
    ids = [str(item["feature_id"]) for item in outputs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate feature IDs in acoustic catalog")
    if any(not value or value == "—" for value in ids):
        raise ValueError("Output features require exact code-facing IDs")


def load_feature_catalog() -> dict[str, Any]:
    matrix = json.loads(_DATA.read_text(encoding="utf-8"))
    if len(matrix["constructs"]) != 79 or len(matrix["families"]) != 13:
        raise ValueError("Master matrix must contain 79 constructs in 13 families")
    for construct in matrix["constructs"]:
        construct["recommended_tasks"] = [key for key, mark in construct["tasks"].items() if mark == "✓"]
        construct["conditional_tasks"] = [key for key, mark in construct["tasks"].items() if mark == "C"]
        construct["use_status"] = USE[construct["matrix_status"]]
        construct["evidence_level"] = None
        construct["evidence_study_count"] = None
        construct["evidence_entry_count"] = None
        construct["outputs"] = []
    # Master matrix has no FEATURE ID(S); the supplied Family 08 document does.
    matrix["outputs"] = []
    matrix["tasks_registry"] = TASKS.copy()
    family01 = json.loads(_FAMILY01.read_text(encoding="utf-8"))
    constructs = {item["construct_id"]: item for item in matrix["constructs"]}
    family01_tasks = {
        "C001": ["sustained_a"], "C002": ["bamboo_passage"],
        "C003": ["sustained_a"], "C004": ["bamboo_passage"],
    }
    outputs01 = []
    for spec in family01["outputs"]:
        construct = constructs[spec["construct_id"]]
        construct["recommended_tasks"] = family01_tasks[spec["construct_id"]]
        construct["conditional_tasks"] = []
        outputs01.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F01", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Research" if spec["construct_id"] == "C004" else "Production",
            "recommended_tasks": family01_tasks[spec["construct_id"]],
            "conditional_tasks": [],
            "qc_range_low": 60 if spec["unit"] == "Hz" else None,
            "qc_range_high": 500 if spec["unit"] == "Hz" else None,
            "qc_range_text": "60–500 Hz" if spec["unit"] == "Hz" else "—",
            "signal_scope": construct["signal_scope"],
            "analysis_region": "stable phonation or final reviewed speech region",
            "estimator": "Praat autocorrelation F0, native sample rate",
            "algorithm": family01["algorithm_version"],
            "algorithm_version": family01["algorithm_version"],
            "default_parameter_set_id": family01["parameter_set_id"],
            "parameter_profile": {"name": "Frozen Praat autocorrelation", "read_only": True,
                                  "active_parameters": {
                                      "pitch_floor_hz": family01["pitch_floor_hz"],
                                      "pitch_ceiling_hz": family01["pitch_ceiling_hz"],
                                      "time_step_sec": family01["time_step_sec"],
                                      "minimum_valid_frames": family01["minimum_valid_frames"],
                                      "minimum_sustained_tracking_yield":
                                          family01["minimum_sustained_tracking_yield"]}},
            "prerequisites": {
                "requires_audio": True, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": False, "requires_task": True,
                "requires_unscaled_audio": False, "requires_voiced_region": True,
                "requires_ddk_events": False, "requires_vowel_tokens": False,
                "requires_prompt_manifest": False},
            "family_spec_approved": True,
            "source_document": family01["source_document"],
        })
    attach_family_outputs(matrix, "F01", outputs01)
    family02 = json.loads(_FAMILY02.read_text(encoding="utf-8"))
    outputs02 = []
    for spec in family02["outputs"]:
        construct = constructs[spec["construct_id"]]
        construct["recommended_tasks"] = ["sustained_a"]
        construct["conditional_tasks"] = []
        outputs02.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F02", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Research" if spec["construct_id"] in {"C008", "C009", "C012"}
                          else "Production",
            "recommended_tasks": ["sustained_a"], "conditional_tasks": [],
            "qc_range_low": None, "qc_range_high": None,
            "qc_range_text": "0–100%" if spec["feature_id"] == "dfp_pct" else "—",
            "signal_scope": construct["signal_scope"],
            "analysis_region": "final reviewed stable phonation region",
            "estimator": "Praat PointProcess periodic cc and Harmonicity cross-correlation",
            "algorithm": family02["algorithm_version"],
            "algorithm_version": family02["algorithm_version"],
            "default_parameter_set_id": family02["parameter_set_id"],
            "parameter_profile": {"name": "Frozen stable-vowel Praat", "read_only": True,
                                  "active_parameters": {
                                      "pitch_floor_hz": 60, "pitch_ceiling_hz": 500,
                                      "period_floor_sec": 0.8 / 500,
                                      "period_ceiling_sec": 1.25 / 60,
                                      "max_period_factor": 1.3,
                                      "max_amplitude_factor": 1.6,
                                      "hnr_time_step_sec": 0.01,
                                      "hnr_periods_per_window": 4.5}},
            "prerequisites": {
                "requires_audio": True, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": False, "requires_task": True,
                "requires_unscaled_audio": False, "requires_voiced_region": True,
                "requires_ddk_events": False, "requires_vowel_tokens": False,
                "requires_prompt_manifest": False},
            "family_spec_approved": True,
            "selectable": spec.get("selectable", True),
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_document": family02["source_document"],
        })
    attach_family_outputs(matrix, "F02", outputs02)
    family04 = json.loads(_FAMILY04.read_text(encoding="utf-8"))
    family05 = json.loads(_FAMILY05.read_text(encoding="utf-8"))
    family04_tasks = {
        "C021": (["bamboo_passage"], ["wstg"]),
        "C022": (["bamboo_passage"], ["wstg"]),
        "C023": (["bamboo_passage"], []),
        "C024": (["bamboo_passage"], []),
        "C025": (["bamboo_passage"], []),
        "C026": (["bamboo_passage"], ["wstg"]),
        "C027": (["bamboo_passage"], []),
        "C028": ([], []),
        "C029": (["bamboo_passage"], []),
        "C030": ([], ["bamboo_passage"]),
        "C031": ([], []),
        "C032": (["bamboo_passage"], []),
    }
    for spec in family04["constructs"]:
        construct = constructs[spec["construct_id"]]
        for field in ("evidence_level", "evidence_study_count", "evidence_entry_count"):
            construct[field] = spec[field]
        construct["recommended_tasks"], construct["conditional_tasks"] = family04_tasks[spec["construct_id"]]
    outputs04 = []
    for spec in family04["outputs"]:
        construct = constructs[spec["construct_id"]]
        granularity = spec["output_granularity"]
        requires_mapping = granularity != "token"
        outputs04.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F04", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Production" if spec.get("selectable", True) else "Blocked",
            "recommended_tasks": construct["recommended_tasks"],
            "conditional_tasks": construct["conditional_tasks"],
            "conditional_reason": ("requires explicitly targeted aligned vowels"
                                   if construct["conditional_tasks"] else ""),
            "evidence_level": construct["evidence_level"],
            "evidence_study_count": construct["evidence_study_count"],
            "evidence_entry_count": construct["evidence_entry_count"],
            "qc_range_low": None, "qc_range_high": None,
            "qc_range_text": ("150–1200 Hz" if spec["feature_id"].startswith("f1_") else
                              "500–3500 Hz" if spec["feature_id"].startswith("f2_token") or
                              spec["feature_id"].startswith("f2_vowel") else
                              "1200–5000 Hz" if spec["feature_id"].startswith("f3_") else
                              "≥0 Hz²" if spec["feature_id"] == "vsa3_iau_hz2" else
                              "≥0 Hz" if spec["feature_id"] == "f2_distance_i_a_hz" else "—"),
            "signal_scope": construct["signal_scope"],
            "analysis_region": ("middle 50% of frozen aligned vowel token"
                                if granularity == "token" else "named frozen-alignment vowel centroids"),
            "estimator": ("Praat/parselmouth Burg; token median"
                          if granularity == "token" else "median token centroids / exact formula"),
            "algorithm": family04["algorithm_version"],
            "algorithm_version": family04["algorithm_version"],
            "default_parameter_set_id": family04["default_parameter_set_id"],
            "parameter_profile": {
                "name": "Frozen native-rate Burg formants", "read_only": True,
                "active_parameters": {"window_length_ms": 25, "time_step_ms": 5,
                                      "max_formants": 5, "pre_emphasis_from_hz": 50,
                                      "formant_ceiling_hz": 5500,
                                      "sensitivity_ceiling_hz": 5000,
                                      "minimum_token_ms": 50,
                                      "minimum_valid_frames": 3}},
            "prerequisites": {"requires_audio": True, "requires_segmentation": True,
                              "requires_final_reviewed_segmentation": True,
                              "requires_alignment": True,
                              "requires_vowel_tokens": True,
                              "requires_vowel_category_mapping": requires_mapping,
                              "requires_task": True,
                              "requires_unscaled_audio": False,
                              "requires_ddk_events": False,
                              "requires_prompt_manifest": False},
            "family_spec_approved": True,
            "selectable": spec.get("selectable", True),
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_document": family04["source_document"],
        })
    attach_family_outputs(matrix, "F04", outputs04)
    for spec in family05["constructs"]:
        construct = constructs[spec["construct_id"]]
        for field in ("evidence_level", "evidence_study_count", "evidence_entry_count"):
            construct[field] = spec[field]
        construct["feature_id_templates"] = spec["feature_id_templates"]
        construct["recommended_tasks"] = []
        construct["conditional_tasks"] = (["bamboo_passage"] if spec["construct_id"] == "C033" else [])
    family06 = json.loads(_FAMILY06.read_text(encoding="utf-8"))
    for spec in family06["constructs"]:
        construct = constructs[spec["construct_id"]]
        for field in ("evidence_level", "evidence_study_count", "evidence_entry_count"):
            construct[field] = spec[field]
        construct["feature_id_templates"] = spec.get("feature_id_templates", [])
        # Detailed target-task contracts supersede the matrix's broad WSTG mark.
        construct["recommended_tasks"] = []
        construct["conditional_tasks"] = (["wstg"] if spec["construct_id"] == "C035" else [])
    outputs06 = []
    for spec in family06["outputs"]:
        construct = constructs[spec["construct_id"]]
        executable = bool(spec["selectable"])
        outputs06.append({
            **spec, "construct_name": construct["construct_name"],
            "family_id": "F06", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Conditional" if executable else "Blocked",
            "recommended_tasks": construct["recommended_tasks"],
            "conditional_tasks": construct["conditional_tasks"],
            "conditional_reason": "Requires licensed source algorithm" if spec["construct_id"] == "C035" else "",
            "evidence_level": construct["evidence_level"],
            "evidence_study_count": construct["evidence_study_count"],
            "evidence_entry_count": construct["evidence_entry_count"],
            "qc_range_low": None, "qc_range_high": None,
            "qc_range_text": construct["qc_range_text"],
            "signal_scope": construct["signal_scope"],
            "analysis_region": ("validated external burst/noise annotation over frozen aligned target"
                                if executable else "source-defined target, unresolved"),
            "estimator": spec["formula"], "algorithm": family06["algorithm_version"],
            "algorithm_version": family06["algorithm_version"],
            "default_parameter_set_id": family06["default_parameter_set_id"],
            "parameter_profile": {"name": "Frozen aligned sub-events", "read_only": True,
                                  "active_parameters": {"m1_window_ms": 20,
                                                        "m1_fft_minimum": 2048,
                                                        "tilt_window_ms": 10,
                                                        "tilt_band_hz": [1500, 5000],
                                                        "wideband_hz": [0, 10000],
                                                        "source_audio_policy": "native rate; no upsampling"}},
            "prerequisites": {"requires_audio": executable,
                              "requires_final_reviewed_segmentation": executable,
                              "requires_alignment": executable,
                              "requires_target_manifest": executable,
                              "requires_acoustic_subevents": executable,
                              "requires_task": executable},
            "family_spec_approved": True, "source_document": family06["source_document"],
        })
    attach_family_outputs(matrix, "F06", outputs06)
    family07 = json.loads(_FAMILY07.read_text(encoding="utf-8"))
    outputs07 = []
    for spec in family07["outputs"]:
        construct = constructs[spec["construct_id"]]
        construct["recommended_tasks"] = spec["recommended_tasks"]
        construct["conditional_tasks"] = spec.get("conditional_tasks", [])
        requires_alignment = bool(spec["requires_alignment"])
        outputs07.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F07", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"], "use_status": "Production",
            "conditional_tasks": spec.get("conditional_tasks", []),
            "qc_range_low": None, "qc_range_high": None,
            "signal_scope": construct["signal_scope"],
            "analysis_region": ("validated aligned token intervals" if requires_alignment else
                                "final reviewed patient timing domain"),
            "estimator": ("aligned token duration statistics" if requires_alignment else
                          "shared reviewed timing representation"),
            "algorithm": family07["algorithm_version"],
            "algorithm_version": family07["algorithm_version"],
            "default_parameter_set_id": (family07["alignment_parameter_set_id"]
                                         if requires_alignment else
                                         family07["timing_parameter_set_id"]),
            "parameter_profile": {
                "name": "Frozen aligned-token profile" if requires_alignment else
                        "Frozen reviewed-timing profile",
                "read_only": True,
                "active_parameters": ({
                    "minimum_alignment_coverage": family07["minimum_alignment_coverage"],
                    "minimum_vowel_duration_sec": family07["minimum_vowel_duration_sec"],
                    "review_vowel_duration_sec": family07["review_vowel_duration_sec"],
                    "sample_sd_ddof": family07["sample_sd_ddof"],
                    "sequence_break_on_manual_exclusion": True,
                } if requires_alignment else {
                    "minimum_internal_pause_ms": 300,
                    "manual_exclusion_policy": "outside analyzable patient timeline",
                    "boundary_source": "frozen reviewed segmentation",
                })},
            "prerequisites": {
                "requires_audio": False, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": requires_alignment, "requires_task": True,
                "requires_unscaled_audio": False, "requires_voiced_region": False,
                "requires_ddk_events": False, "requires_vowel_tokens":
                    spec["construct_id"] in {"C046", "C047"},
                "requires_prompt_manifest": False},
            "family_spec_approved": True,
            "selectable": spec.get("selectable", True),
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_document": family07["source_document"],
        })
    attach_family_outputs(matrix, "F07", outputs07)
    family08 = json.loads(_FAMILY08.read_text(encoding="utf-8"))
    for construct_id in ("C048", "C049"):
        # Detailed family task applicability supersedes the broader matrix cells.
        constructs[construct_id]["recommended_tasks"] = ["bamboo_passage"]
        constructs[construct_id]["conditional_tasks"] = []
    outputs = []
    for spec in family08["outputs"]:
        construct = constructs[spec["construct_id"]]
        outputs.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F08",
            "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "source_use_status": construct["matrix_status"],
            "use_status": "Production",
            "recommended_tasks": ["bamboo_passage"],
            "conditional_tasks": [],
            "qc_range_low": 0,
            "qc_range_high": 10,
            "qc_range_text": "0–10 syllables/s" if spec["unit"] == "syllables/s" else "≥0 words/min",
            "analysis_unit": "Whole utterance/prompt",
            "signal_scope": construct["signal_scope"],
            "analysis_region": "first to last final patient-speech boundary",
            "estimator": "versioned prompt count / reviewed task timing",
            "algorithm": family08["algorithm_version"],
            "algorithm_version": family08["algorithm_version"],
            "default_parameter_set_id": family08["parameter_set_id"],
            "parameter_profile": {
                "name": "Validated default", "read_only": True,
                "active_parameters": {"pause_min_sec": family08["pause_min_sec"],
                                      "boundary_source": "frozen reviewed segmentation"}},
            "prerequisites": {
                "requires_audio": False,
                "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": False,
                "requires_task": True,
                "requires_unscaled_audio": False,
                "requires_voiced_region": False,
                "requires_ddk_events": False,
                "requires_vowel_tokens": False,
                "requires_prompt_manifest": True},
            "family_spec_approved": True,
            "selectable": True,
            "source_document": family08["source_document"],
        })
    attach_family_outputs(matrix, "F08", outputs)
    family09 = json.loads(_FAMILY09.read_text(encoding="utf-8"))
    for construct_id in (f"C{number:03d}" for number in range(50, 58)):
        constructs[construct_id]["recommended_tasks"] = ["bamboo_passage"]
        constructs[construct_id]["conditional_tasks"] = []
    outputs09 = []
    for spec in family09["outputs"]:
        construct = constructs[spec["construct_id"]]
        research = spec.get("research_only", False)
        outputs09.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F09",
            "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "source_use_status": construct["matrix_status"],
            "use_status": "Research" if research else "Production",
            "recommended_tasks": ["bamboo_passage"],
            "conditional_tasks": [],
            "qc_range_low": None, "qc_range_high": None,
            "signal_scope": construct["signal_scope"],
            "analysis_region": "reviewed patient utterance and eligible internal pause/phrase events",
            "estimator": "shared reviewed pause/phrase event table",
            "algorithm": family09["algorithm_version"],
            "algorithm_version": family09["algorithm_version"],
            "default_parameter_set_id": family09["parameter_set_id"],
            "parameter_profile": {
                "name": "Validated default" if not research else "Research components",
                "read_only": True,
                "active_parameters": {
                    "minimum_internal_pause_ms": family09["minimum_internal_pause_ms"],
                    "sample_sd_ddof": family09["sample_sd_ddof"],
                    "boundary_source": "frozen reviewed segmentation"}},
            "prerequisites": {
                "requires_audio": False, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": False, "requires_task": True,
                "requires_unscaled_audio": False, "requires_voiced_region": False,
                "requires_ddk_events": False, "requires_vowel_tokens": False,
                "requires_prompt_manifest": False},
            "family_spec_approved": True,
            "selectable": spec.get("selectable", True),
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_document": family09["source_document"],
        })
    attach_family_outputs(matrix, "F09", outputs09)
    family10 = json.loads(_FAMILY10.read_text(encoding="utf-8"))
    for construct_id in ("C058", "C059"):
        constructs[construct_id]["recommended_tasks"] = ["ddk"]
        constructs[construct_id]["conditional_tasks"] = []
    outputs10 = []
    for spec in family10["outputs"]:
        construct = constructs[spec["construct_id"]]
        outputs10.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F10", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "source_use_status": construct["matrix_status"],
            "use_status": "Production", "recommended_tasks": ["ddk"],
            "conditional_tasks": [], "qc_range_low": 0, "qc_range_high": None,
            "signal_scope": construct["signal_scope"],
            "analysis_region": "final reviewed DDK analysis domain",
            "estimator": "final reviewed DDK events and midpoints",
            "algorithm": family10["algorithm_version"],
            "algorithm_version": family10["algorithm_version"],
            "default_parameter_set_id": family10["parameter_set_id"],
            "parameter_profile": {
                "name": "Validated default", "read_only": True,
                "active_parameters": {
                    "minimum_valid_events": family10["minimum_valid_events"],
                    "event_landmark": family10["event_landmark"],
                    "rate_denominator": family10["rate_denominator"],
                    "sequence_break_on_manual_exclusion": True,
                    "temporal_variability_definition":
                        family10["temporal_variability_definition"],
                }},
            "prerequisites": {
                "requires_audio": False, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": False, "requires_task": True,
                "requires_unscaled_audio": False, "requires_voiced_region": False,
                "requires_ddk_events": True, "requires_vowel_tokens": False,
                "requires_prompt_manifest": False},
            "family_spec_approved": True, "selectable": True,
            "gui_selectable": False,
            "source_document": family10["source_document"],
        })
    attach_family_outputs(matrix, "F10", outputs10)
    family11 = json.loads(_FAMILY11.read_text(encoding="utf-8"))
    outputs11 = []
    for spec in family11["outputs"]:
        construct = constructs[spec["construct_id"]]
        outputs11.append({
            **spec, "construct_name": construct["construct_name"],
            "family_id": "F11", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"], "use_status": "Research",
            "recommended_tasks": construct["recommended_tasks"],
            "conditional_tasks": construct["conditional_tasks"],
            "qc_range_low": 0 if spec["feature_id"] == "absolute_energy_fs2" else None,
            "qc_range_high": None, "qc_range_text": spec.get("range", "—"),
            "signal_scope": construct["signal_scope"],
            "analysis_region": "final reviewed patient analysis region",
            "estimator": "NumPy/SciPy native-rate digital waveform statistics",
            "algorithm": family11["algorithm_version"],
            "algorithm_version": family11["algorithm_version"],
            "default_parameter_set_id": family11["parameter_set_id"],
            "parameter_profile": {"name": "Frozen native-amplitude profile", "read_only": True,
                                  "active_parameters": {"sd_ddof": 1, "skew_bias": False,
                                      "kurtosis_fisher": True, "kurtosis_bias": False,
                                      "normalized_direct_amplitude_policy": "NaN"}},
            "prerequisites": {"requires_audio": True, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True, "requires_alignment": False,
                "requires_task": True, "requires_unscaled_audio": spec["feature_id"] in {
                    "absolute_energy_fs2", "sound_power_digital", "amp_mean_fs", "amp_sd_fs",
                    "amp_min_fs", "amp_max_fs"}, "requires_voiced_region": False,
                "requires_ddk_events": False, "requires_vowel_tokens": False,
                "requires_prompt_manifest": False},
            "family_spec_approved": True, "selectable": spec.get("selectable", True),
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_document": family11["source_document"],
        })
    attach_family_outputs(matrix, "F11", outputs11)
    family12 = json.loads(_FAMILY12.read_text(encoding="utf-8"))
    mfcc_specs = []
    for coefficient in range(1, 14):
        for statistic in ("mean", "sd"):
            mfcc_specs.append({
                "construct_id": "C065", "feature_id": f"mfcc{coefficient:02d}_{statistic}",
                "human_name": f"MFCC {coefficient:02d} — {statistic.upper()}",
                "unit": "cepstral coefficient", "evidence_level": "NOT ESTABLISHED",
                "evidence_study_count": 3, "evidence_entry_count": 3,
                "analysis_unit": "Frames",
            })
    contrast_specs = [{
        "construct_id": "C067", "feature_id": f"spectral_contrast_band{band}_db",
        "human_name": f"Spectral contrast — band {band}", "unit": "dB",
        "evidence_level": "NOT ESTABLISHED", "evidence_study_count": 1,
        "evidence_entry_count": 1, "analysis_unit": "Frames",
    } for band in range(7)]
    outputs12 = []
    for spec in [*mfcc_specs, *family12["outputs"], *contrast_specs]:
        construct = constructs[spec["construct_id"]]
        is_mfcc = spec["construct_id"] == "C065"
        outputs12.append({
            **spec, "construct_name": construct["construct_name"],
            "formula": spec.get("formula", "Frozen explicit Family 12 estimator"),
            "family_id": "F12", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Production" if spec.get("selectable", True) else "Research",
            "recommended_tasks": construct["recommended_tasks"],
            "conditional_tasks": construct["conditional_tasks"],
            "qc_range_low": 0 if spec["feature_id"].startswith(("zcr_", "spectral_bandwidth")) else None,
            "qc_range_high": 1 if spec["feature_id"].startswith("zcr_") else None,
            "qc_range_text": "0–1" if spec["feature_id"].startswith("zcr_") else "—",
            "signal_scope": construct["signal_scope"],
            "analysis_region": "final reviewed patient speech/task region",
            "estimator": "explicit private 48 kHz STFT/log-Mel transform",
            "algorithm": family12["algorithm_version"],
            "algorithm_version": family12["algorithm_version"],
            "default_parameter_set_id": (family12["mfcc_parameter_set_id"] if is_mfcc else
                                         family12["spectral_parameter_set_id"]),
            "parameter_profile": {"name": "Frozen 48 kHz private spectral profile",
                "read_only": True, "active_parameters": {
                    "working_sample_rate_hz": family12["working_sample_rate_hz"],
                    "window_ms": family12["window_ms"], "hop_ms": family12["hop_ms"],
                    "n_fft": family12["n_fft"], "center": False,
                    "minimum_frames": family12["minimum_frames"],
                    **(family12["mfcc"] if is_mfcc else {}),
                    **(family12["spectral_contrast"]
                       if spec["construct_id"] == "C067" else {})}},
            "prerequisites": {"requires_audio": True, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True, "requires_alignment": False,
                "requires_task": True, "requires_unscaled_audio": False,
                "requires_voiced_region": False, "requires_ddk_events": False,
                "requires_vowel_tokens": False, "requires_prompt_manifest": False},
            "family_spec_approved": True, "selectable": spec.get("selectable", True),
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_document": family12["source_document"],
        })
    attach_family_outputs(matrix, "F12", outputs12)
    family13 = json.loads(_FAMILY13.read_text(encoding="utf-8"))
    outputs13 = []
    for spec in family13["constructs"]:
        construct = constructs[spec["construct_id"]]
        for field in ("evidence_level", "evidence_study_count", "evidence_entry_count"):
            construct[field] = spec[field]
        construct["recommended_tasks"] = spec["recommended_tasks"]
        construct["conditional_tasks"] = []
        construct["family_spec_approved"] = True
        construct["source_status"] = spec["status"]
        construct["source_ids"] = spec.get("source_ids", [])
        construct["source_feature_templates"] = spec.get("source_templates", [])
        construct["source_limitation"] = spec["missing_definition"]
        construct["scientific_meaning"] = spec["interpretation"]
        construct["analysis_region"] = spec["representation"]
        construct["analysis_unit"] = spec["analysis_unit"]
        construct["unit"] = spec["unit"]
        construct["qc_range_text"] = spec["range"]
        construct["formula"] = spec["formula"]
        construct["source_document"] = family13["source_document"]
        for feature_id in spec.get("source_ids", []):
            is_factor = feature_id.endswith("factor_if_frozen")
            outputs13.append({
                "construct_id": spec["construct_id"],
                "feature_id": feature_id,
                "human_name": spec["source_output_names"][feature_id],
                "construct_name": construct["construct_name"],
                "family_id": "F13", "family_name": construct["family_name"],
                "matrix_status": construct["matrix_status"],
                "use_status": "Research",
                "recommended_tasks": spec["recommended_tasks"],
                "conditional_tasks": [],
                "evidence_level": spec["evidence_level"],
                "evidence_study_count": spec["evidence_study_count"],
                "evidence_entry_count": spec["evidence_entry_count"],
                "unit": "z-score" if is_factor else "bits" if feature_id ==
                        "ppe_source_replication_only" else "source-specific",
                "qc_range_low": None, "qc_range_high": None,
                "qc_range_text": "0–log2(31) bits" if feature_id ==
                                 "ppe_source_replication_only" else "—",
                "signal_scope": construct["signal_scope"],
                "analysis_unit": spec["analysis_unit"],
                "analysis_region": spec["representation"],
                "scientific_meaning": spec["interpretation"],
                "formula": spec["formula"],
                "estimator": "Source procedure not reproducibly frozen",
                "algorithm": "not_implemented",
                "algorithm_version": "not_implemented",
                "default_parameter_set_id": "not_frozen",
                "parameter_profile": {"name": "No frozen source profile", "read_only": True,
                                      "active_parameters": {}},
                "prerequisites": {},
                "family_spec_approved": True,
                "selectable": False,
                "source_status": "NOT_IMPLEMENTED",
                "blocked_reason": spec["missing_definition"],
                "source_document": family13["source_document"],
            })
    attach_family_outputs(matrix, "F13", outputs13)
    return matrix


def attach_family_outputs(catalog: dict[str, Any], family_id: str,
                          outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach exact family-spec leaves without changing construct identities."""
    constructs = {item["construct_id"]: item for item in catalog["constructs"]}
    validate_feature_ids([*catalog["outputs"], *outputs])
    for output in outputs:
        construct = constructs.get(output["construct_id"])
        if construct is None or construct["family_id"] != family_id:
            raise ValueError("Output must reference an existing construct in its family")
        if not output.get("family_spec_approved"):
            raise ValueError("Output requires family-spec approval before registration")
        for field in ("evidence_level", "evidence_study_count", "evidence_entry_count"):
            if output.get(field) is None and construct.get(field) is not None:
                output[field] = construct[field]
    for output in outputs:
        constructs[output["construct_id"]]["outputs"].append(output)
    catalog["outputs"].extend(outputs)
    for construct in (c for c in catalog["constructs"] if c["family_id"] == family_id):
        leaves = construct["outputs"]
        for field in ("evidence_level", "evidence_study_count", "evidence_entry_count"):
            values = {leaf.get(field) for leaf in leaves}
            if len(values) == 1:
                construct[field] = values.pop()
    return catalog


def task_key(task_name: str) -> str | None:
    """Resolve only registered display names; never infer task semantics."""
    normalized = task_name.strip().casefold()
    return next((key for key, label in TASKS.items() if label.casefold() == normalized), None)


def task_fit(item: dict[str, Any], task_name: str, task_type: str | None = None) -> str:
    """Use only explicit exact-task or source-approved task-type mappings."""
    key = task_key(task_name)
    if key is not None and key in item.get("recommended_tasks", ()):
        return "Recommended"
    if key is not None and key in item.get("conditional_tasks", ()):
        return "Conditional"
    if item.get("family_spec_approved") and not item.get("explicit_task_type_compatibility"):
        return "Not specified"
    if task_type and task_type in item.get("recommended_task_types", ()):
        return "Recommended"
    if task_type and task_type in item.get("conditional_task_types", ()):
        return "Conditional"
    return "Not specified"


def task_indicator(item: dict[str, Any], task_name: str, *,
                   task_type: str | None = None) -> tuple[str, str]:
    """Task fit is independent of code and current-run prerequisites."""
    task_label = task_name.strip() or "current task"
    fit = task_fit(item, task_name, task_type)
    if fit == "Recommended":
        if item.get("conditional_reason"):
            return "AMBER", f"Conditional for {task_label} — {item['conditional_reason']}"
        return "GREEN", f"Recommended for {task_label}."
    if fit == "Conditional":
        reason = item.get("conditional_reason")
        suffix = f" — {reason}" if reason else "."
        return "AMBER", f"Conditional for {task_label}{suffix}"
    return "GRAY", f"Not specifically recommended for {task_label}."


def feature_availability(item: dict[str, Any], *,
                         prerequisite_problems: list[str] | None = None) -> tuple[str, str]:
    """Report registered implementation, independent of task and run inputs."""
    if (item.get("feature_id")
            and item.get("family_id") in REGISTERED_EXECUTORS
            and item.get("matrix_status") != "BLOCKED"
            and not item.get("blocked_reason")
            and item.get("family_spec_approved")
            and item.get("selectable")
            and item.get("algorithm_version")
            and item.get("default_parameter_set_id")):
        return "Implemented", "Registered exact output calculator."
    if item.get("blocked_reason"):
        return "Not implemented", str(item["blocked_reason"])
    return "Not implemented", "No approved executable leaf is registered."


def prerequisite_issues(output: dict[str, Any], *, task_name: str,
                        final_decisions: Path | None, final_intervals: Path | None,
                        alignment_available: bool = False,
                        prompt_available: bool = False,
                        vowel_mapping_available: bool = False,
                        target_manifest_available: bool = False,
                        acoustic_subevents_available: bool = False) -> list[str]:
    """Schema reserved for approved exact outputs from future family documents."""
    if not output.get("family_spec_approved", False):
        return ["Exact output algorithm awaits its family specification"]
    requires = output["prerequisites"]
    issues = []
    if requires.get("requires_task") and not task_name.strip():
        issues.append("Task name is missing")
    if requires.get("requires_final_reviewed_segmentation") and not (
            final_decisions and final_decisions.is_file() and final_intervals and final_intervals.is_file()):
        issues.append("Final reviewed segmentation is unavailable")
    if requires.get("requires_ddk_events") and final_decisions and final_decisions.is_file():
        try:
            with final_decisions.open(encoding="utf-8-sig", newline="") as stream:
                has_ddk = any(
                    row.get("segmentation_method") == "ddk_energy"
                    and row.get("final_decision") in {"KEEP_AUTO", "KEEP_MANUAL"}
                    for row in csv.DictReader(stream))
        except (OSError, csv.Error):
            has_ddk = False
        if not has_ddk:
            issues.append("Frozen reviewed DDK events are unavailable")
    if requires.get("requires_alignment") and not alignment_available:
        issues.append("Required vowel/phone alignment is unavailable")
    if requires.get("requires_vowel_category_mapping") and not vowel_mapping_available:
        issues.append("Versioned vowel-category mapping is unavailable")
    if requires.get("requires_prompt_manifest") and not prompt_available:
        issues.append("Prompt numerator is unavailable")
    if requires.get("requires_target_manifest") and not target_manifest_available:
        issues.append("Approved phonetic target definition is unavailable")
    if requires.get("requires_acoustic_subevents") and not acoustic_subevents_available:
        issues.append("Validated acoustic sub-event annotations are unavailable")
    return issues
