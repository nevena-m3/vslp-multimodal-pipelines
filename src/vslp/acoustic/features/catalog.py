"""Source-bound registry for the 79 master-matrix constructs and approved additions.

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
_FAMILY03 = Path(__file__).with_name("data") / "family03.json"
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
    "ddk_pa": "/pa/", "ddk_ta": "/ta/",
    "ddk_ka": "/ka/", "ddk_pataka": "/pataka/",
    "wstg": "WSTG",
    "sentence_s5": "S5",
    "sentence_s2": "S2",
    "sentence_s3": "S3",
    "sentence_s6": "S6",
    "sentence_s7": "S7",
    "sentence_s9": "S9",
}
REGISTERED_EXECUTORS = frozenset({"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F13",
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
    family03 = json.loads(_FAMILY03.read_text(encoding="utf-8"))
    active03 = set(family03["active_construct_ids"])
    for spec in family03["constructs"]:
        construct = constructs[spec["construct_id"]]
        construct.update({
            "evidence_level": "LIMITED", "evidence_study_count": 1,
            "evidence_entry_count": spec["evidence_entry_count"],
            "recommended_tasks": ["sustained_a"], "conditional_tasks": [],
            "family_spec_approved": True, "source_status": "NOT_IMPLEMENTED",
            "id_provenance": "NO_PUBLIC_ID", "source_placeholder": "blocked_proprietary",
            "source_limitation": spec["blocked_reason"],
            "scientific_meaning": spec["source_terminology"],
            "analysis_region": "stable sustained phonation",
            "analysis_unit": "vendor-defined glottal cycle phase or point",
            "unit": spec["unit"], "qc_range_text": spec["range"],
            "formula": "No reproducible source algorithm is disclosed.",
            "source_document": family03["source_document"],
            "active_in_gui": spec["construct_id"] in active03,
        })
    names03 = {
        "C013": "Glottal closure / phonatory effort",
        "C014": "Glottal cycle phase quotients",
        "C016": "Amplitude tremor",
        "C020": "Vibratory instability / blockage",
    }
    for construct_id, name in names03.items():
        constructs[construct_id].update(
            construct_name=name, matrix_status="PROD", use_status="Production",
            source_status="OPEN_VSLP_METHOD; VCS parity not claimed",
            source_limitation="Open analogue; proprietary VCS Pr parity not claimed.",
            source_placeholder="", id_provenance="SOURCE_DEFINED_ID",
            recommended_tasks=["sustained_a"], conditional_tasks=[],
            analysis_region="centered reviewed stable /a/ <=2 s",
            analysis_unit="glottal cycles or amplitude modulation within trial",
            unit="varies by output", formula="See explicit open VSLP child outputs.",
            qc_range_text="No clinical range established",
        )
    outputs03 = []
    recommendation03 = {"sustained_vowel": True, "sentence:S5": False,
                        "sentence:S6": False, "bamboo_passage": False,
                        "sentence:other": False, "ddk": False}
    for spec in family03["open_outputs"]:
        construct = constructs[spec["construct_id"]]
        selectable = spec.get("selectable", True)
        outputs03.append({
            **spec, "construct_name": construct["construct_name"],
            "family_id": "F03", "family_name": construct["family_name"],
            "matrix_status": "PROD", "use_status": "Production" if selectable else "Research",
            "evidence_level": "LIMITED", "evidence_study_count": 1,
            "evidence_entry_count": construct["evidence_entry_count"],
            "recommended_tasks": ["sustained_a"] if selectable else [],
            "conditional_tasks": [],
            "recommended_for_task": recommendation03 if selectable else {},
            "executable_for_task": {key: bool(selectable and key == "sustained_vowel")
                                    for key in recommendation03},
            "qc_range_low": None, "qc_range_high": None,
            "qc_range_text": "0–1" if spec["feature_id"] in {
                "glottal_open_quotient", "glottal_closed_quotient",
                "glottal_closing_quotient"} else "—",
            "signal_scope": "Speech · Cycles",
            "analysis_region": "centered reviewed stable /a/ <=2 s",
            "analysis_unit": "reviewed trial; cycles or modulation frames",
            "estimator": "QCP-lineage GIF or 50-Hz Hilbert amplitude modulation",
            "algorithm": family03["algorithm_version"],
            "algorithm_version": family03["algorithm_version"],
            "default_parameter_set_id": family03["parameter_set_id"],
            "parameter_profile": {"name": family03["parameter_set_id"],
                                  "read_only": True},
            "source_derived_parameters": {}, "vslp_replication_parameters": {},
            "source_parity_status": "pending",
            "implementation_class": spec["method_class"],
            "id_provenance": "PROJECT_DEFINED_ID",
            "identity_basis": "Open VSLP/source-lineage output; not a proprietary VCS Pr value",
            "prerequisites": {
                "requires_audio": True, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": False, "requires_task": True,
                "requires_unscaled_audio": False, "requires_voiced_region": True,
                "requires_ddk_events": False, "requires_vowel_tokens": False,
                "requires_prompt_manifest": False},
            "family_spec_approved": True, "selectable": selectable,
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_equivalent": False,
            "source_document": family03["source_document"],
        })
    attach_family_outputs(matrix, "F03", outputs03)
    family01_tasks = {
        construct_id: [task for task, key in (("sustained_a", "sustained_vowel"),
                                             ("sentence_s5", "sentence:S5"),
                                             ("bamboo_passage", "bamboo_passage"))
                       if recommendations.get(key)]
        for construct_id, recommendations in family01["task_recommendations"].items()
    }
    outputs01 = []
    for spec in family01["outputs"]:
        construct = constructs[spec["construct_id"]]
        if spec["construct_id"] == "C004":
            construct.update(family01["intonation_construct"])
        leaf_recommendations = spec.get(
            "recommended_for_task", family01["task_recommendations"][spec["construct_id"]])
        leaf_tasks = [task for task, key in (("sustained_a", "sustained_vowel"),
                                              ("sentence_s5", "sentence:S5"),
                                              ("bamboo_passage", "bamboo_passage"))
                      if leaf_recommendations.get(key)]
        construct["recommended_tasks"] = family01_tasks[spec["construct_id"]]
        construct["conditional_tasks"] = []
        construct["recommended_for_task"] = family01["task_recommendations"][spec["construct_id"]]
        outputs01.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F01", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Research" if spec["construct_id"] == "C004" else "Production",
            "recommended_tasks": (leaf_tasks
                                  if spec["selectable"] else []),
            "conditional_tasks": [],
            "executable_for_task": bool(spec["selectable"]),
            "recommended_for_task": (leaf_recommendations
                                     if spec["selectable"] else {}),
            "stable_region_policy_id": family01["stable_region_policy_id"],
            "interpretation_note": spec.get("interpretation_note", ""),
            "deprecated_alias_of": spec.get("deprecated_alias_of", ""),
            "construct_role": spec.get("construct_role", ""),
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
    for addition in family02.get("project_constructs", []):
        if addition["construct_id"] in constructs:
            raise ValueError("Duplicate project construct ID")
        construct = {
            **constructs["C012"], **addition, "outputs": [],
            "recommended_tasks": ["sustained_a"], "conditional_tasks": [],
            "family_spec_approved": True,
        }
        matrix["constructs"].append(construct)
        constructs[addition["construct_id"]] = construct
    outputs02 = []
    for spec in family02["outputs"]:
        construct = constructs[spec["construct_id"]]
        recommendations = spec.get("recommended_for_task", family02[
            "task_recommendations"][spec["construct_id"]])
        task_keys = {"sustained_vowel": "sustained_a", "sentence:S5": "sentence_s5",
                     "sentence:S6": "sentence_s6", "bamboo_passage": "bamboo_passage",
                     "ddk": "ddk"}
        recommended_tasks = [task_keys[key] for key, enabled in recommendations.items()
                             if enabled and key in task_keys]
        profile_name = ({"C008": "gne_vslp_replication_v1",
                         "C009": "pvi_vslp_replication_v1",
                         "C010": "cpp_praat_unsmoothed_v1",
                         "C012": "harmonic_vslp_replication_v1",
                         "C080": "h1h2_iseli_2007_v1"}.get(spec["construct_id"]))
        if spec["feature_id"] == "cpps_db":
            profile_name = None
        parameter_values = (family02["replication_profiles"][profile_name]
                            if profile_name else {
                                "pitch_floor_hz": 60, "pitch_ceiling_hz": 500,
                                "period_floor_sec": 0.8 / 500,
                                "period_ceiling_sec": 1.25 / 60,
                                "max_period_factor": 1.3,
                                "max_amplitude_factor": 1.6,
                                "hnr_time_step_sec": 0.01,
                                "hnr_periods_per_window": 4.5})
        lineage = {"C008": "gne", "C009": "pvi", "C012": "harmonic",
                   "C080": "h1h2"}.get(
            spec["construct_id"])
        provenance = family02.get("source_parameter_provenance", {})
        source_parameters = {key: parameter_values[key] for key in provenance.get(
            f"{lineage}_source", []) if key in parameter_values}
        vslp_parameters = {key: parameter_values[key] for key in provenance.get(
            f"{lineage}_vslp", []) if key in parameter_values}
        construct["recommended_for_task"] = family02["task_recommendations"][spec[
            "construct_id"]]
        construct["recommended_tasks"] = recommended_tasks
        construct["conditional_tasks"] = []
        outputs02.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F02", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Research" if spec["construct_id"] in {"C008", "C009", "C012"}
                          else "Production",
            "recommended_tasks": recommended_tasks, "conditional_tasks": [],
            "recommended_for_task": recommendations,
            "executable_for_task": {key: bool(spec.get("selectable", True))
                                    for key in recommendations},
            "qc_range_low": None, "qc_range_high": None,
            "qc_range_text": "0–100%" if spec["feature_id"] == "dfp_pct" else "—",
            "signal_scope": construct["signal_scope"],
            "analysis_region": "centered reviewed stable vowel or eligible voiced trial islands",
            "estimator": spec.get("estimator", "Praat PointProcess periodic cc and Harmonicity cross-correlation"),
            "algorithm": spec.get("algorithm_version", family02["algorithm_version"]),
            "algorithm_version": spec.get("algorithm_version", family02["algorithm_version"]),
            "default_parameter_set_id": profile_name or family02["parameter_set_id"],
            "parameter_profile": {"name": profile_name or "Frozen Praat perturbation/HNR",
                                  "read_only": True,
                                  "active_parameters": parameter_values},
            "source_derived_parameters": source_parameters,
            "vslp_replication_parameters": vslp_parameters,
            "source_parity_status": spec.get("source_parity_status",
                                              "reference_validation_pending"),
            "implementation_class": spec.get("implementation_class", "production_candidate"),
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
            "source_document": spec.get("source_document", family02["source_document"]),
        })
    attach_family_outputs(matrix, "F02", outputs02)
    family04 = json.loads(_FAMILY04.read_text(encoding="utf-8"))
    family05 = json.loads(_FAMILY05.read_text(encoding="utf-8"))
    family04_tasks = {
        "C021": (["sustained_a", "sentence_s5", "bamboo_passage"], []),
        "C022": (["sentence_s5", "bamboo_passage"], []),
        "C023": (["bamboo_passage"], []),
        "C024": (["bamboo_passage"], []),
        "C025": (["bamboo_passage"], []),
        "C026": (["wstg"], []),
        "C027": (["bamboo_passage"], []),
        "C028": (["sentence_s3"], []),
        "C029": (["bamboo_passage"], []),
        "C030": (["sentence_s3"], []),
        "C031": (["sentence_s3"], []),
        "C032": (["bamboo_passage"], []),
    }
    for spec in family04["constructs"]:
        construct = constructs[spec["construct_id"]]
        for field in ("evidence_level", "evidence_study_count", "evidence_entry_count"):
            construct[field] = spec[field]
        construct["recommended_tasks"], construct["conditional_tasks"] = family04_tasks[spec["construct_id"]]
    outputs04 = []
    f04_project_ids = {"f1_median_hz", "f2_median_hz", "f3_median_hz",
                       "vowel_distance_ai_hz", "vowel_distance_au_hz",
                       "vowel_distance_iu_hz"} | {
                           spec["feature_id"] for spec in family04["outputs"]
                           if spec["construct_id"] == "C032"}
    for spec in family04["outputs"]:
        construct = constructs[spec["construct_id"]]
        granularity = spec["output_granularity"]
        requires_mapping = False  # AA/IY/UW mapping is frozen in the task registry.
        recommendations = spec.get("recommended_tasks", construct["recommended_tasks"])
        outputs04.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F04", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Production" if spec.get("selectable", True) else "Audit only",
            "recommended_tasks": recommendations,
            "conditional_tasks": construct["conditional_tasks"],
            "conditional_reason": ("requires explicitly targeted aligned vowels"
                                   if construct["conditional_tasks"] else ""),
            "evidence_level": construct["evidence_level"],
            "evidence_study_count": construct["evidence_study_count"],
            "evidence_entry_count": construct["evidence_entry_count"],
            "qc_range_low": None, "qc_range_high": None,
            "qc_range_text": ("150–1200 Hz" if spec["feature_id"].startswith("f1_") else
                              "500–3500 Hz" if spec["feature_id"].startswith("f2_token") or
                              spec["feature_id"].startswith("f2_vowel") or
                              spec["feature_id"] == "f2_median_hz" else
                              "1200–5000 Hz" if spec["feature_id"].startswith("f3_") else
                              "≥0 Hz²" if spec["feature_id"] == "vsa3_iau_hz2" else
                              "≥0 Hz" if spec["feature_id"] == "f2_distance_i_a_hz" else "—"),
            "signal_scope": construct["signal_scope"],
            "analysis_region": ("middle 50% of frozen aligned vowel token"
                                if granularity == "token" else
                                "reviewed /a/ midpoint or aligned vowel-token medians"
                                if spec["feature_id"] in {"f1_median_hz", "f2_median_hz", "f3_median_hz"}
                                else "named frozen-alignment vowel centroids"),
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
                              "alignment_exempt_tasks": ["sustained_a"] if spec["feature_id"]
                              in {"f1_median_hz", "f2_median_hz", "f3_median_hz"} else [],
                              "requires_vowel_tokens": True,
                              "requires_vowel_category_mapping": requires_mapping,
                              "requires_task": True,
                              "requires_unscaled_audio": False,
                              "requires_ddk_events": False,
                              "requires_prompt_manifest": False},
            "family_spec_approved": True,
            "selectable": spec.get("selectable", True),
            "method_class": ("VSLP_TASK_ADAPTATION" if spec["feature_id"] == "f3_median_hz"
                             else spec.get("method_class", "STANDARD_SOURCE_FROZEN")),
            "active_in_gui": granularity == "recording",
            "id_provenance": ("PROJECT_DEFINED_ID" if spec["feature_id"] in f04_project_ids
                              else "SOURCE_DEFINED_ID"),
            "source_parity_status": "pending" if spec["construct_id"] == "C028"
                                    else "implementation_tested_reference_pending",
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_document": family04["source_document"],
        })
    attach_family_outputs(matrix, "F04", outputs04)
    for spec in family05["constructs"]:
        construct = constructs[spec["construct_id"]]
        for field in ("evidence_level", "evidence_study_count", "evidence_entry_count"):
            construct[field] = spec[field]
        construct["historical_feature_id_templates"] = spec["feature_id_templates"]
        construct["feature_id_templates"] = []
        construct["recommended_tasks"] = (["sentence_s2", "sentence_s5"]
                                          if spec["construct_id"] == "C033" else ["sentence_s2"])
        construct["conditional_tasks"] = []
    outputs05 = []
    for spec in family05["outputs"]:
        construct = constructs[spec["construct_id"]]
        outputs05.append({
            **spec, "family_id": "F05", "family_name": "Formant dynamics",
            "construct_name": construct["construct_name"],
            "matrix_status": "PROD", "use_status": "Production",
            "evidence_level": construct["evidence_level"],
            "evidence_study_count": construct["evidence_study_count"],
            "evidence_entry_count": construct["evidence_entry_count"],
            "recommended_tasks": [spec["task_id"]], "conditional_tasks": [],
            "recommended_for_task": {"sentence_s2": spec["task_id"] == "sentence_s2",
                                     "sentence_s5": spec["task_id"] == "sentence_s5"},
            "executable_for_task": {"sentence_s2": spec["task_id"] == "sentence_s2",
                                    "sentence_s5": spec["task_id"] == "sentence_s5"},
            "qc_range_low": None, "qc_range_high": None,
            "qc_range_text": "No clinical range established",
            "signal_scope": "Speech · Aligned diphthong token",
            "analysis_region": "central 20–80% of final target phone",
            "analysis_unit": "token → trial median → recording median",
            "output_granularity": "recording",
            "formula": ("signed robust fitted slope" if spec["construct_id"] == "C033"
                        else "absolute fitted endpoint difference"),
            "estimator": "F04 Praat Burg frames + bisquare IRLS",
            "algorithm": family05["algorithm_version"],
            "algorithm_version": family05["algorithm_version"],
            "default_parameter_set_id": family05["parameter_set_id"],
            "parameter_profile": {"name": family05["parameter_set_id"], "read_only": True,
                                  "active_parameters": {"formant_profile_id": family05["formant_profile_id"],
                                                        "analysis_window": [0.2, 0.8],
                                                        "minimum_frames": 5}},
            "prerequisites": {"requires_audio": True, "requires_segmentation": True,
                              "requires_final_reviewed_segmentation": True,
                              "requires_alignment": True, "requires_task": True,
                              "requires_vowel_tokens": True,
                              "requires_ddk_events": False,
                              "requires_prompt_manifest": False},
            "family_spec_approved": True, "selectable": True,
            "active_in_gui": True, "blocked_reason": "",
            "method_class": ("VSLP_TASK_ADAPTATION" if spec["task_id"] == "sentence_s5"
                             else "SOURCE_LINEAGE"),
            "source_parity_status": "pending",
            "source_document": family05["source_document"],
        })
    attach_family_outputs(matrix, "F05", outputs05)
    family06 = json.loads(_FAMILY06.read_text(encoding="utf-8"))
    for spec in family06["constructs"]:
        construct = constructs[spec["construct_id"]]
        for field in ("evidence_level", "evidence_study_count", "evidence_entry_count"):
            construct[field] = spec[field]
        construct["feature_id_templates"] = spec.get("feature_id_templates", [])
        # Detailed target-task contracts supersede the matrix's broad WSTG mark.
        construct["recommended_tasks"] = {
            "C036": ["wstg"], "C037": ["sentence_s3"],
            "C038": ["wstg"], "C039": ["sentence_s7"],
            "C042": ["sentence_s9"],
        }.get(spec["construct_id"], [])
        construct["conditional_tasks"] = []
    outputs06 = []
    for spec in family06["outputs"]:
        construct = constructs[spec["construct_id"]]
        executable = bool(spec["selectable"])
        outputs06.append({
            **spec, "construct_name": construct["construct_name"],
            "family_id": "F06", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Conditional" if executable else "Blocked",
            "recommended_tasks": ([spec["task_id"]] if executable else []),
            "conditional_tasks": construct["conditional_tasks"],
            "conditional_reason": "Requires licensed source algorithm" if spec["construct_id"] == "C035" else "",
            "evidence_level": construct["evidence_level"],
            "evidence_study_count": construct["evidence_study_count"],
            "evidence_entry_count": construct["evidence_entry_count"],
            "qc_range_low": None, "qc_range_high": None,
            "qc_range_text": construct["qc_range_text"],
            "signal_scope": construct["signal_scope"],
            "analysis_region": ("versioned native acoustic subevent within Final Alignment V1 phone"
                                if executable else "historical inactive construct"),
            "estimator": spec["formula"], "algorithm": family06["algorithm_version"],
            "algorithm_version": family06["algorithm_version"],
            "default_parameter_set_id": family06["default_parameter_set_id"],
            "parameter_profile": {"name": "VSLP native acoustic subevents v2", "read_only": True,
                                  "active_parameters": {"m1_window_ms": 20,
                                                        "m1_fft_minimum": 2048,
                                                        "tilt_window_ms": 10,
                                                        "tilt_band_hz": [1500, 5000],
                                                        "band_a_hz": [720, 10000, 4200, 10000],
                                                        "band_b_hz": [720, 6500, 720, 2150],
                                                        "source_audio_policy": "native rate; no upsampling"}},
            "prerequisites": {"requires_audio": executable,
                              "requires_final_reviewed_segmentation": executable,
                              "requires_alignment": executable,
                              "requires_target_manifest": False,
                              "requires_acoustic_subevents": False,
                              "requires_task": executable},
            "family_spec_approved": True, "source_document": family06["source_document"],
        })
    attach_family_outputs(matrix, "F06", outputs06)
    family07 = json.loads(_FAMILY07.read_text(encoding="utf-8"))
    constructs["C043"]["active_in_gui"] = False
    constructs["C043"]["recommended_tasks"] = []
    constructs["C046"]["construct_name"] = "Vocalic duration"
    for construct_id, label, parent in (
        ("VSLP_F07_S3_SENTENCE_DURATION", "S3 sentence duration", "C045"),
        ("VSLP_F07_S3_VOCALIC_PROPORTION", "S3 vocalic proportion", "C046"),
    ):
        addition = {**constructs[parent], "construct_id": construct_id,
                    "construct_name": label, "outputs": [], "active_in_gui": True,
                    "id_provenance": "PROJECT_DEFINED_ID",
                    "recommended_tasks": ["sentence_s3"], "conditional_tasks": []}
        matrix["constructs"].append(addition)
        constructs[construct_id] = addition
    outputs07 = []
    for spec in family07["outputs"]:
        construct = constructs[spec["construct_id"]]
        construct["recommended_tasks"] = spec["recommended_tasks"]
        construct["conditional_tasks"] = []
        requires_alignment = bool(spec["requires_alignment"])
        outputs07.append({
            **spec,
            "construct_name": construct["construct_name"],
            "family_id": "F07", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Production" if spec["selectable"] else "Blocked",
            "evidence_level": construct.get("evidence_level") or "NOT ESTABLISHED",
            "evidence_study_count": construct.get("evidence_study_count", 0),
            "evidence_entry_count": construct.get("evidence_entry_count", 0),
            "analysis_unit": "trial then recording",
            "active_in_gui": bool(spec["selectable"]),
            "conditional_tasks": [],
            "qc_range_low": None, "qc_range_high": None,
            "signal_scope": construct["signal_scope"],
            "analysis_region": ("Final Alignment V1 canonical vowel intervals" if requires_alignment else
                                "final reviewed timing and confirmed trial domain"),
            "estimator": ("canonical vowel duration statistics" if requires_alignment else
                          "frozen reviewed speech/trial boundaries"),
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
                    "minimum_vowel_duration_sec": family07["minimum_vowel_duration_sec"],
                    "review_vowel_duration_sec": family07["review_vowel_duration_sec"],
                    "sample_sd_ddof": family07["sample_sd_ddof"],
                    "minimum_bamboo_vowel_count": family07["minimum_bamboo_vowel_count"],
                    "sequence_break_on_manual_exclusion": True,
                } if requires_alignment else {
                    "manual_exclusion_policy": "hard exclusion is never patient speech",
                    "boundary_source": "frozen reviewed segmentation",
                })},
            "prerequisites": {
                "requires_audio": False, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": requires_alignment, "requires_task": True,
                "requires_unscaled_audio": False, "requires_voiced_region": False,
                "requires_ddk_events": False, "requires_vowel_tokens":
                    spec["construct_id"] in {"C046", "C047", "VSLP_F07_S3_VOCALIC_PROPORTION"},
                "requires_prompt_manifest": False},
            "family_spec_approved": True,
            "selectable": spec.get("selectable", True),
            "blocked_reason": spec.get("blocked_reason", ""),
            "id_provenance": ("PROJECT_DEFINED_ID" if spec["construct_id"].startswith("VSLP_")
                              else "SOURCE_DEFINED_ID"),
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
            "qc_range_high": None,
            "qc_range_text": ">0 words/min",
            "analysis_unit": "Bamboo trial, then recording median",
            "signal_scope": construct["signal_scope"],
            "analysis_region": "F07 frozen reviewed Bamboo timing",
            "estimator": "60 × registered scientific word count / F07 reviewed duration",
            "algorithm": family08["algorithm_version"],
            "algorithm_version": family08["algorithm_version"],
            "default_parameter_set_id": family08["parameter_set_id"],
            "parameter_profile": {
                "name": "Frozen nominal Bamboo WPM", "read_only": True,
                "active_parameters": {"boundary_source": "frozen reviewed segmentation",
                                      "word_count_source": "expected_prompt_count"}},
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
                "requires_prompt_manifest": False},
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
            "use_status": "Production",
            "recommended_tasks": ["ddk", f"ddk_{spec['target']}"],
            "conditional_tasks": [], "qc_range_low": 0, "qc_range_high": None,
            "signal_scope": construct["signal_scope"],
            "analysis_region": "first ten accepted reviewed cycles within one trial",
            "estimator": "reviewed onset/offset rate and center-interval variability",
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
                "requires_audio": False, "requires_segmentation": False,
                "requires_final_reviewed_segmentation": True,
                "requires_alignment": False, "requires_task": True,
                "requires_unscaled_audio": False, "requires_voiced_region": False,
                "requires_ddk_events": True, "requires_vowel_tokens": False,
                "requires_prompt_manifest": False},
            "family_spec_approved": True, "selectable": True,
            "gui_selectable": True,
            "source_document": family10["source_document"],
            "id_provenance": "PROJECT_DEFINED_ID",
            "identity_basis": "Frozen F10 task-specific reviewed-event output",
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
            "recommended_tasks": [spec["task_id"]],
            "conditional_tasks": [],
            "qc_range_low": 0 if spec["construct_id"] in {"C060", "C061", "C063"} else None,
            "qc_range_high": None, "qc_range_text": spec.get("range", "—"),
            "signal_scope": construct["signal_scope"],
            "analysis_region": spec["analysis_unit"],
            "estimator": "Native-waveform NumPy/SciPy; private 48-kHz polyphase copy for C060/C061",
            "algorithm": family11["algorithm_version"],
            "algorithm_version": family11["algorithm_version"],
            "default_parameter_set_id": family11["parameter_set_id"],
            "parameter_profile": {"name": "Frozen F11 reviewed-region profile", "read_only": True,
                                  "active_parameters": {"sd_ddof": 1, "skew_bias": False,
                                      "kurtosis_fisher": True, "kurtosis_bias": False,
                                      "energy_power_analysis_rate_hz": 48000,
                                      "visibility_window_sec": 0.050}},
            "prerequisites": {"requires_audio": True, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True, "requires_alignment": False,
                "requires_task": True, "requires_unscaled_audio": False,
                "requires_voiced_region": False,
                "requires_ddk_events": False, "requires_vowel_tokens": False,
                "requires_prompt_manifest": False},
            "family_spec_approved": True, "selectable": spec.get("selectable", True),
            "gui_selectable": True,
            "id_provenance": "PROJECT_DEFINED_ID",
            "identity_basis": "Frozen F11 task-specific reviewed-region output",
            "blocked_reason": spec.get("blocked_reason", ""),
            "source_document": family11["source_document"],
        })
    attach_family_outputs(matrix, "F11", outputs11)
    from vslp.acoustic.features.family12_v2 import VERSION as f12_version, specs as f12_specs

    family12 = json.loads(_FAMILY12.read_text(encoding="utf-8"))
    outputs12 = []
    for spec in f12_specs():
        construct = constructs[spec["construct_id"]]
        is_mfcc = spec["construct_id"] == "C065"
        task_id = spec["task_id"]
        outputs12.append({
            **spec, "construct_name": construct["construct_name"],
            "evidence_level": "LIMITED" if spec["construct_id"] in {"C070", "C071"}
                              else "NOT ESTABLISHED",
            "evidence_study_count": 1, "evidence_entry_count": 1,
            "analysis_unit": "Frames",
            "formula": "Frozen task-specific Family 12 estimator",
            "family_id": "F12", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"],
            "use_status": "Production" if spec["selectable"] else "Research",
            "recommended_tasks": [task_id] if spec["selectable"] else [],
            "conditional_tasks": [],
            "qc_range_low": 0 if spec["construct_id"] in {"C066", "C069"} else None,
            "qc_range_high": 1 if spec["construct_id"] == "C069" else None,
            "qc_range_text": "0–1" if spec["construct_id"] == "C069" else "—",
            "signal_scope": construct["signal_scope"],
            "analysis_region": ("centered reviewed stable phonation <=2 s"
                                if task_id == "sustained_a" else
                                "independent eligible reviewed speech intervals"),
            "estimator": "private 48 kHz center=False Hann STFT",
            "algorithm": f12_version, "algorithm_version": f12_version,
            "default_parameter_set_id": (family12["mfcc_parameter_set_id"] if is_mfcc else
                                         family12["spectral_parameter_set_id"]),
            "parameter_profile": {"name": "Frozen 48 kHz private F12 profile",
                "read_only": True, "active_parameters": {
                    "working_sample_rate_hz": 48000, "window_ms": 25, "hop_ms": 10,
                    "n_fft": 2048, "center": False, "minimum_frames": 10,
                    **(family12["mfcc"] if is_mfcc else {}),
                    **(family12["spectral_contrast"]
                       if spec["construct_id"] == "C067" else {})}},
            "prerequisites": {"requires_audio": True, "requires_segmentation": True,
                "requires_final_reviewed_segmentation": True, "requires_alignment": False,
                "requires_task": True, "requires_unscaled_audio": False,
                "requires_voiced_region": False, "requires_ddk_events": False,
                "requires_vowel_tokens": False, "requires_prompt_manifest": False},
            "family_spec_approved": True,
            "blocked_reason": spec["blocked_reason"],
            "source_document": family12["source_document"],
            "id_provenance": "PROJECT_DEFINED_ID",
            "identity_basis": "Frozen task-specific F12 output",
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
    outputs13.append({**outputs13[0],
                      "feature_id": "vowel_sample_entropy_pending_definition",
                      "human_name": "Sample entropy — source definition pending",
                      "construct_id": "C073", "construct_name": constructs["C073"]["construct_name"],
                      "selectable": False, "source_status": "RESEARCH_PENDING_SOURCE_DEFINITION",
                      "blocked_reason": "research_pending_source_definition",
                      "id_provenance": "PROJECT_DEFINED_ID",
                      "identity_basis": "Documented nonselectable candidate"})
    attach_family_outputs(matrix, "F13", outputs13)
    from vslp.acoustic.features.family13_v1 import VERSION as f13_version, specs as f13_specs

    active13 = []
    for spec in f13_specs():
        construct = constructs[spec["construct_id"]]
        task_id = spec["task_id"]
        active13.append({
            **spec, "construct_name": construct["construct_name"],
            "family_id": "F13", "family_name": construct["family_name"],
            "matrix_status": construct["matrix_status"], "use_status": "Production",
            "recommended_tasks": [task_id], "conditional_tasks": [],
            "evidence_level": construct["evidence_level"],
            "evidence_study_count": construct["evidence_study_count"],
            "evidence_entry_count": construct["evidence_entry_count"],
            "analysis_unit": "sentence" if task_id == "bamboo_passage" else "trial",
            "analysis_region": "canonical aligned sentence, reviewed speech" if task_id ==
                               "bamboo_passage" else "centered reviewed stable phonation <=2 s"
                               if task_id == "sustained_a" else "reviewed trial speech",
            "scientific_meaning": spec["human_name"],
            "formula": "Frozen Family 13 operational component",
            "estimator": f13_version, "algorithm": f13_version,
            "algorithm_version": f13_version,
            "default_parameter_set_id": f13_version,
            "parameter_profile": {"name": f13_version, "read_only": True,
                                  "active_parameters": {}},
            "prerequisites": {"requires_audio": True, "requires_segmentation": True,
                              "requires_final_reviewed_segmentation": True,
                              "requires_alignment": task_id == "bamboo_passage",
                              "requires_task": True, "requires_unscaled_audio": False,
                              "requires_voiced_region": spec["construct_id"] == "C072",
                              "requires_ddk_events": False, "requires_vowel_tokens": False,
                              "requires_prompt_manifest": False},
            "family_spec_approved": True, "source_document": "family13_scientific_review.md",
            "id_provenance": "PROJECT_DEFINED_ID",
            "identity_basis": "Frozen F13 operational component",
            "qc_range_low": 0, "qc_range_high": None,
            "qc_range_text": "component-specific", "signal_scope": construct["signal_scope"],
        })
    attach_family_outputs(matrix, "F13", active13)
    for output in matrix["outputs"]:
        is_template = "<" in output["feature_id"] or ">" in output["feature_id"]
        output["id_provenance"] = output.get(
            "id_provenance", "PROJECT_TEMPLATE" if is_template else "SOURCE_DEFINED_ID")
        output["identity_basis"] = output.get("identity_basis") or (
            "Template written in detailed DOCX FEATURE ID(S) field" if is_template
            else "Concrete ID specified by detailed DOCX FEATURE ID(S) field")
        if output["family_id"] == "F12" and (
                output["feature_id"].startswith("mfcc") or
                output["feature_id"].startswith("spectral_contrast_band")):
            output["identity_basis"] = "Deterministic expansion of explicit DOCX FEATURE ID(S) range"
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
    # Older Acoustic Setup saved the explicit Bamboo task label as its ID.
    if normalized == "bamboo":
        return "bamboo_passage"
    return next((key for key, label in TASKS.items() if label.casefold() == normalized), None)


def project_task_key(manifest: dict[str, Any]) -> str:
    """Resolve explicit Setup task identity, including its older Bamboo label."""
    value = str(manifest.get("task_id") or "").strip()
    if value in TASKS:
        return value
    return task_key(value) or task_key(str(manifest.get("task_name") or "")) or value


def supports_task(output: dict[str, Any], task_name: str,
                  task_type: str | None = None) -> bool:
    """Check execution routing, independently of recommendation and selection."""
    key = task_name if task_name in TASKS else task_key(task_name)
    if key is None:
        return False
    executable = output.get("executable_for_task")
    if isinstance(executable, dict):
        group = ("sustained_vowel" if key == "sustained_a" else
                 "bamboo_passage" if key == "bamboo_passage" else
                 "ddk" if key == "ddk" or key.startswith("ddk_") else
                 f"sentence:{key.rsplit('_', 1)[-1].upper()}")
        return bool(executable.get(key, executable.get(group,
                    executable.get(task_type, executable.get("sentence:other", False)))))
    if executable is not None:
        return bool(executable)
    if output.get("task_id"):
        return key == output["task_id"]
    family, feature_id = output.get("family_id"), str(output.get("feature_id", ""))
    if family == "F04":
        if feature_id == "f2_median_hz":
            return key in {"sustained_a", "sentence_s5", "bamboo_passage"}
        if feature_id == "f1_median_hz":
            return key in {"sentence_s5", "bamboo_passage"}
        if feature_id == "f3_median_hz":
            return key == "wstg"
        if feature_id.startswith("vowel_") or feature_id == "f2_distance_i_a_hz":
            return key == "sentence_s3" if feature_id.startswith((
                "vowel_distance_", "vowel_envelope_")) or feature_id == (
                "f2_distance_i_a_hz") else key == "bamboo_passage"
        return key == "bamboo_passage"
    if family == "F07":
        return key == ("sentence_s3" if feature_id.startswith("s3_") else
                       "bamboo_passage")
    if family in {"F08", "F09"}:
        return key == "bamboo_passage"
    if family == "F10":
        target = feature_id.split("_", 2)[1]
        return key in {"ddk", f"ddk_{target}"}
    # Older implementations without an explicit route decide applicability in
    # their own stage. Recommendation metadata is never an execution gate.
    return True


def requires_alignment_for_task(output: dict[str, Any], task_name: str) -> bool:
    """A sustained midpoint can use reviewed segmentation without word/phone tiers."""
    requirement = output.get("prerequisites", {})
    return bool(requirement.get("requires_alignment") and task_key(task_name)
                not in requirement.get("alignment_exempt_tasks", ()))


def task_fit(item: dict[str, Any], task_name: str, task_type: str | None = None) -> str:
    """Use only explicit exact-task or source-approved task-type mappings."""
    if item.get("family_id") in {"F01", "F02"} and "recommended_for_task" in item:
        from vslp.acoustic.features.family01_task import recommendation_key, task_context

        context = task_context({"task_name": task_name, "task_id": task_key(task_name),
                                "task_type": task_type})
        return ("Recommended" if item["recommended_for_task"].get(
            recommendation_key(context), False) else "Not specified")
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
                        acoustic_subevents_available: bool = False,
                        reviewed_ddk_events_available: bool = False) -> list[str]:
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
    if requires.get("requires_ddk_events") and not reviewed_ddk_events_available:
        issues.append("Reviewed DDK events required")
    if requires_alignment_for_task(output, task_name) and not alignment_available:
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
