"""Family 06 definitions and guarded target selection."""

import json

import numpy as np
import pandas as pd
import pytest

from vslp.acoustic.features.catalog import feature_availability, load_feature_catalog
from vslp.acoustic.features.family06 import (
    IMPLEMENTED_IDS, TEMPLATES, first_spectral_moment, load_target_manifest,
    load_validated_subevents, matched_m1_contrast, noise_duration,
    normalized_duration_contrast, stop_burst_tilt, wideband_noise_energy,
)
from vslp.acoustic.features.family06_stage import _measure_target


def _target(role, index):
    return {"target_id": f"pair1_{role}", "feature_id": "m1_t_minus_k_hz",
            "word": f"controlled_{role}", "phone": role.upper(),
            "word_index": index, "phone_index": 1, "role": role, "pair_id": "pair1",
            "acoustic_region_type": "post_burst_20ms",
            "pairing_rule": "matched_pair_id",
            "aggregation": "mean_target_moments_then_difference"}


def test_eight_constructs_exact_ids_and_unresolved_templates():
    catalog = load_feature_catalog()
    constructs = [c for c in catalog["constructs"] if c["family_id"] == "F06"]
    leaves = [o for o in catalog["outputs"] if o["family_id"] == "F06"]
    assert len(constructs) == 8
    assert {o["feature_id"] for o in leaves} == IMPLEMENTED_IDS | {
        "blocked_aural_analytics_ap", "band_noise_contrast_variant1",
        "band_noise_contrast_variant2"}
    assert all(feature_availability(o)[0] == "Implemented"
               for o in leaves if o["feature_id"] in IMPLEMENTED_IDS)
    assert all(feature_availability(o)[0] == "Not implemented"
               for o in leaves if o["feature_id"] not in IMPLEMENTED_IDS)
    assert not any("gop" in o["feature_id"].lower() for o in leaves)
    assert not any(template in {o["feature_id"] for o in leaves} for template in TEMPLATES)
    assert all(c["evidence_level"] for c in constructs)


def test_family06_gui_leaf_contract():
    import os
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    window = AcousticPipelineWindow()
    for feature_id in IMPLEMENTED_IDS:
        item = window.feature_items[feature_id]
        assert item.text(1) == feature_id
        assert item.text(3) == "Implemented"
        assert item.text(4) in {"LIMITED", "NOT ESTABLISHED"}
        assert item.flags() & Qt.ItemIsUserCheckable
    for feature_id in ("blocked_aural_analytics_ap", "band_noise_contrast_variant1",
                       "band_noise_contrast_variant2"):
        item = window.feature_items[feature_id]
        assert item.text(3) == "Not implemented"
        assert not item.flags() & Qt.ItemIsUserCheckable
    window.close()


def test_target_manifest_requires_explicit_complete_t_k_pair(tmp_path):
    path = tmp_path / "targets.json"
    data = {"manifest_version": "1", "task_id": "wstg", "targets": [_target("t", 1)]}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_target_manifest(path, "wstg")[1] == "invalid_target_definition"
    data["targets"].append(_target("k", 2))
    path.write_text(json.dumps(data), encoding="utf-8")
    targets, reason = load_target_manifest(path, "wstg")
    assert reason == "" and len(targets) == 2
    assert load_target_manifest(path, "bamboo_passage")[1] == "target_manifest_task_mismatch"
    data["targets"][0]["phone"] = "K"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_target_manifest(path, "wstg")[1] == "invalid_target_definition"
    assert load_target_manifest(None, "wstg")[1] == "missing_target_definition"


def test_approved_original_time_subevents_only(tmp_path):
    path = tmp_path / "annotations.csv"
    row = {"recording_id": "r1", "target_id": "pair1_t", "word_index": 1,
           "phone_index": 1, "annotation_source": "manual", "reviewer": "reviewer",
           "approved": True, "time_axis": "original_recording_seconds",
           "burst_start_sec": .2}
    pd.DataFrame([row]).to_csv(path, index=False)
    table, reason = load_validated_subevents(path)
    assert reason == "" and len(table) == 1
    pd.DataFrame([{**row, "approved": False}]).to_csv(path, index=False)
    assert load_validated_subevents(path)[1] == "unapproved_acoustic_sub_event"
    pd.DataFrame([{**row, "time_axis": "cropped_seconds"}]).to_csv(path, index=False)
    assert load_validated_subevents(path)[1] == "invalid_acoustic_sub_event_time_axis"
    pd.DataFrame([{**row, "annotation_source": ""}]).to_csv(path, index=False)
    assert load_validated_subevents(path)[1] == "missing_acoustic_sub_event_provenance"


def test_first_moment_and_matched_pair_hand_calculation():
    sr = 48000
    t = np.arange(sr) / sr
    low = np.sin(2 * np.pi * 2000 * t)
    high = np.sin(2 * np.pi * 4000 * t)
    assert first_spectral_moment(high, sr, .2) > first_spectral_moment(low, sr, .2)
    rows = pd.DataFrame([
        {"pair_id": "p1", "role": "t", "token_value": 4000.},
        {"pair_id": "p1", "role": "k", "token_value": 2000.},
        {"pair_id": "p2", "role": "t", "token_value": 4500.},
        {"pair_id": "p2", "role": "k", "token_value": 2500.},
    ])
    assert matched_m1_contrast(rows) == (2000., "")
    assert matched_m1_contrast(rows.iloc[:1])[1] == "missing_matched_t_k_targets"


def test_wideband_native_bandwidth_and_digital_scaling():
    sr = 48000
    t = np.arange(sr) / sr
    signal = .2 * np.sin(2 * np.pi * 3000 * t)
    energy = wideband_noise_energy(signal, sr, .1, .2)
    assert energy > 0
    assert wideband_noise_energy(signal * 2, sr, .1, .2) == pytest.approx(energy * 4)
    for low_rate in (16000, 20000):
        with pytest.raises(ValueError, match="insufficient_bandwidth"):
            wideband_noise_energy(np.zeros(low_rate), low_rate, .1, .2)


def test_burst_tilt_and_internal_duration_formulae():
    sr = 48000
    t = np.arange(sr) / sr
    low = np.sin(2 * np.pi * 1700 * t)
    high = np.sin(2 * np.pi * 4700 * t)
    assert stop_burst_tilt(low, sr, .2) < stop_burst_tilt(high, sr, .2)
    assert noise_duration(.2, .5) == pytest.approx(.3)
    assert normalized_duration_contrast(.3, .1) == pytest.approx(1.)
    with pytest.raises(ValueError, match="invalid_duration_pair"):
        normalized_duration_contrast(0, .1)


def test_phone_bounds_do_not_substitute_for_acoustic_landmark():
    sr = 48000
    t = np.arange(sr) / sr
    audio = .2 * np.sin(2 * np.pi * 3000 * t)
    phone = pd.Series({"start_sec": .2, "end_sec": .5})
    target = {"feature_id": "stop_burst_spectral_tilt_db_khz"}
    with pytest.raises(ValueError, match="missing_validated_burst_start"):
        _measure_target(target, pd.Series({"annotation_source": "manual"}), phone,
                        audio, sr, 0, 1, pd.DataFrame())
    annotation = pd.Series({"burst_start_sec": .3})
    value, start, end = _measure_target(target, annotation, phone,
                                        audio, sr, 0, 1, pd.DataFrame())
    assert np.isfinite(value) and (start, end) == pytest.approx((.3, .31))
    excluded = pd.DataFrame([{"start_sec": .305, "end_sec": .32}])
    with pytest.raises(ValueError, match="sub_event_overlaps_manual_exclusion"):
        _measure_target(target, annotation, phone, audio, sr, 0, 1, excluded)


def test_wideband_target_requires_approved_noise_interval_and_source_bandwidth():
    phone = pd.Series({"start_sec": .2, "end_sec": .5})
    target = {"feature_id": "wideband_noise_energy_0_10khz"}
    annotation = pd.Series({"noise_start_sec": .25, "noise_end_sec": .35})
    with pytest.raises(ValueError, match="insufficient_bandwidth"):
        _measure_target(target, annotation, phone, np.zeros(16000), 16000,
                        0, 1, pd.DataFrame())
    with pytest.raises(ValueError, match="missing_validated_noise_interval"):
        _measure_target(target, pd.Series({"burst_start_sec": .25}), phone,
                        np.zeros(48000), 48000, 0, 1, pd.DataFrame())
