import pandas as pd

from vslp.analysis.kinematics.features import (
    CANONICAL_FEATURE_IDS,
    feature_manifest_dataframe,
    write_features_only_exports,
)


def test_write_features_only_exports_creates_canonical_and_manifest(tmp_path):
    root = tmp_path
    df = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "task_guess": ["open", "speech"],
            "status": ["ok", "qc_flagged"],
            "feature_qc_flags": ["", "low_face_detection"],
            "face_detected_fraction": [0.99, 0.72],
            "path_vert_med": [0.10, 0.20],
            "rom_vert_med": [0.30, 0.40],
            "mouth_aperture_mean": [0.11, 0.21],
            "source_path": ["a.mp4", "b.mp4"],
        }
    )
    paths = write_features_only_exports(df, root)

    canonical = pd.read_csv(paths["canonical65_csv"])
    only = pd.read_csv(paths["features_only_csv"])
    ml_ready = pd.read_csv(paths["ml_ready_csv"])
    manifest = pd.read_csv(paths["feature_manifest_csv"])

    assert "video_id" in canonical.columns
    assert "status" in canonical.columns
    assert "path_vert_med" in canonical.columns
    assert "rom_vert_med" in canonical.columns
    assert list(only.columns) == ["path_vert_med", "rom_vert_med"]
    assert "video_id" in ml_ready.columns
    assert "path_vert_med" in ml_ready.columns
    assert "status" not in ml_ready.columns
    roles = dict(zip(manifest["column_name"], manifest["column_role"]))
    assert roles["path_vert_med"] == "canonical_feature"
    assert roles["mouth_aperture_mean"] == "dense_engineering_summary"
    assert roles["source_path"] == "metadata"
    assert paths["n_canonical_features_expected"] == len(CANONICAL_FEATURE_IDS)
    assert paths["n_canonical_features_present"] == 2


def test_feature_manifest_defaults_to_canonical_features():
    manifest = feature_manifest_dataframe()
    assert len(manifest) == len(CANONICAL_FEATURE_IDS)
    assert set(manifest["column_role"]) == {"canonical_feature"}
    assert manifest["include_in_ml_default"].all()
