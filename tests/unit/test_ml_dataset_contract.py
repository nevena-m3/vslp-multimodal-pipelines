from pathlib import Path
import json

import pandas as pd

from vslp.ml.dataset_contract import MLDatasetContractConfig, build_ml_datasets


def _write(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def test_build_ml_datasets_creates_unimodal_and_early_fusion_tables(tmp_path: Path):
    acoustic = pd.DataFrame({
        "subject_id": ["S1", "S2", "S3"],
        "session_id": ["V1", "V1", "V1"],
        "task": ["passage", "passage", "passage"],
        "speech_rate": [110.0, 150.0, 160.0],
        "jitter": [0.5, 0.2, 0.3],
    })
    kinematic = pd.DataFrame({
        "subject_id": ["S1", "S2"],
        "session_id": ["V1", "V1"],
        "task": ["passage", "passage"],
        "path_vert_med": [0.2, 0.4],
        "sLL_vert_med": [1.1, 1.5],
    })
    metadata = pd.DataFrame({
        "subject_id": ["S1", "S2", "S3"],
        "session_id": ["V1", "V1", "V1"],
        "task": ["passage", "passage", "passage"],
        "diagnosis": ["ALS", "Control", "ALS"],
    })
    aman = pd.DataFrame({
        "column_name": ["speech_rate", "jitter", "subject_id"],
        "include_in_ml_default": [True, True, False],
        "column_role": ["canonical_feature", "canonical_feature", "metadata"],
        "modality": ["acoustic", "acoustic", "acoustic"],
    })
    kman = pd.DataFrame({
        "column_name": ["path_vert_med", "sLL_vert_med", "subject_id"],
        "include_in_ml_default": [True, True, False],
        "column_role": ["canonical_feature", "canonical_feature", "metadata"],
        "modality": ["kinematic", "kinematic", "kinematic"],
    })
    cfg = MLDatasetContractConfig(
        acoustic_features_csv=_write(tmp_path / "a.csv", acoustic),
        kinematic_features_csv=_write(tmp_path / "k.csv", kinematic),
        metadata_csv=_write(tmp_path / "m.csv", metadata),
        acoustic_manifest_csv=_write(tmp_path / "aman.csv", aman),
        kinematic_manifest_csv=_write(tmp_path / "kman.csv", kman),
        target_column="diagnosis",
    )
    res = build_ml_datasets(tmp_path, cfg)
    early = pd.read_csv(res["early_fusion_csv"])
    overlap = pd.read_csv(res["modality_overlap_csv"])
    manifest = json.loads(Path(res["dataset_manifest_json"]).read_text(encoding="utf-8"))
    assert len(early) == 2
    assert "acoustic__speech_rate" in early.columns
    assert "kinematic__path_vert_med" in early.columns
    assert "diagnosis" in early.columns
    assert int(overlap.loc[overlap["overlap_group"] == "both_modalities", "n_rows"].iloc[0]) == 2
    assert manifest["problem_type"] == "binary_classification"
    assert manifest["leakage_precheck"]["early_fusion"]["status"] == "pass"


def test_matched_subject_mode_restricts_unimodal_tables(tmp_path: Path):
    acoustic = pd.DataFrame({"subject_id": ["S1", "S2", "S3"], "task": ["a", "a", "a"], "speech_rate": [1, 2, 3]})
    kinematic = pd.DataFrame({"subject_id": ["S1", "S2"], "task": ["a", "a"], "path_vert_med": [4, 5]})
    cfg = MLDatasetContractConfig(
        acoustic_features_csv=_write(tmp_path / "a.csv", acoustic),
        kinematic_features_csv=_write(tmp_path / "k.csv", kinematic),
        comparison_mode="matched_subjects",
    )
    res = build_ml_datasets(tmp_path, cfg)
    a = pd.read_csv(res["acoustic_only_csv"])
    assert set(a["subject_id"]) == {"S1", "S2"}
