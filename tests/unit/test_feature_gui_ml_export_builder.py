from pathlib import Path

import pandas as pd

from vslp.analysis.features.ml_export_builder import build_ml_export_package, classify_feature_table_columns


def test_classifies_acoustic_registry_features_and_suppresses_provenance():
    df = pd.DataFrame({
        "subject_id": ["S1"],
        "diagnosis": ["ALS"],
        "source_file_path": ["a.wav"],
        "speech_rate": [120.0],
        "jitter_local": [0.02],
        "feature_extraction_status": ["ok"],
    })
    registry = pd.DataFrame({
        "feature": ["speech_rate", "jitter_local"],
        "subsystem": ["respiratory_timing", "phonatory"],
        "unit": ["words/min", "%"],
        "meaning": ["speech rate", "cycle perturbation"],
    })
    manifest = classify_feature_table_columns(df, "acoustic", registry)
    roles = dict(zip(manifest["column"], manifest["role"]))
    include = dict(zip(manifest["column"], manifest["include_in_ml_default"]))
    assert roles["subject_id"] == "id"
    assert roles["diagnosis"] == "target_candidate"
    assert roles["source_file_path"] == "provenance"
    assert roles["feature_extraction_status"] == "qc_metric"
    assert include["speech_rate"] is True
    assert include["jitter_local"] is True


def test_builds_feature_gui_ml_export_package_with_early_fusion(tmp_path: Path):
    acoustic = pd.DataFrame({
        "subject_id": ["S1", "S2"],
        "session_id": ["V1", "V1"],
        "task": ["bamboo", "bamboo"],
        "diagnosis": ["ALS", "Control"],
        "speech_rate": [100.0, 140.0],
        "cpp_mean": [12.0, 16.0],
        "source_file_path": ["a.wav", "b.wav"],
    })
    acoustic_path = tmp_path / "acoustic_features_per_file.csv"
    acoustic.to_csv(acoustic_path, index=False)
    registry = pd.DataFrame({
        "feature": ["speech_rate", "cpp_mean"],
        "subsystem": ["respiratory_timing", "phonatory_voice_quality"],
        "unit": ["words/min", "dB"],
        "meaning": ["speech rate", "cepstral peak prominence"],
        "evidence_tier": ["A", "B"],
    })
    registry_path = tmp_path / "selected_acoustic_feature_registry.csv"
    registry.to_csv(registry_path, index=False)
    kinematic = pd.DataFrame({
        "subject_id": ["S1", "S2"],
        "session_id": ["V1", "V1"],
        "task": ["bamboo", "bamboo"],
        "video_id": ["v1", "v2"],
        "face_detected_fraction": [1.0, 0.95],
        "mouth_aperture_median": [0.3, 0.4],
        "mouth_aperture_iqr": [0.05, 0.06],
        "status": ["ok", "ok"],
    })
    kinematic_path = tmp_path / "kinematic_aggregated_features.csv"
    kinematic.to_csv(kinematic_path, index=False)

    result = build_ml_export_package(
        output_root=tmp_path / "out",
        acoustic_features=acoustic_path,
        acoustic_registry=registry_path,
        kinematic_features=kinematic_path,
    )
    tables = result.tables
    assert (result.output_dir / "tables" / "acoustic_ml_ready.csv").exists()
    assert (result.output_dir / "tables" / "kinematic_ml_ready.csv").exists()
    assert (result.output_dir / "tables" / "multimodal_early_fusion_ml_ready.csv").exists()
    assert "speech_rate" in tables["acoustic_features_only"].columns
    assert "mouth_aperture_median" in tables["kinematic_features_only"].columns
    assert "face_detected_fraction" not in tables["kinematic_features_only"].columns
    assert len(tables["multimodal_early_fusion_ml_ready"]) == 2
    fusion_cols = set(tables["multimodal_early_fusion_ml_ready"].columns)
    assert "acoustic__speech_rate" in fusion_cols
    assert "kinematic__mouth_aperture_median" in fusion_cols
    assert result.summary["early_fusion_rows"] == 2
