from pathlib import Path

import numpy as np
import pandas as pd

from vslp.analysis.kinematics.features import FeatureComputationConfig, compute_feature_timeseries, run_feature_computation


def _norm_table(n=40):
    t = np.arange(n) * 33.333
    phase = np.linspace(0, 4 * np.pi, n)
    aperture = 0.2 + 0.08 * (np.sin(phase) + 1.0)
    df = pd.DataFrame({"frame": np.arange(n), "timestamp_ms": t, "face_detected": True})
    # center/nose
    for idx in [1, 13, 14, 61, 291, 78, 308, 152]:
        df[f"{idx}_x_norm"] = 0.0
        df[f"{idx}_y_norm"] = 0.0
        df[f"{idx}_z_norm"] = 0.0
    df["13_y_norm"] = -aperture / 2
    df["14_y_norm"] = aperture / 2
    df["61_x_norm"] = -0.4
    df["291_x_norm"] = 0.4
    df["78_x_norm"] = -0.25
    df["308_x_norm"] = 0.25
    df["152_y_norm"] = 0.9 + 0.02 * np.sin(phase)
    return df


def test_compute_feature_timeseries_contains_geometry_and_velocity():
    ts, meta = compute_feature_timeseries(_norm_table(), FeatureComputationConfig(use_smoothed_signals=False))
    assert "mouth_aperture" in ts.columns
    assert "mouth_aperture_velocity" in ts.columns
    assert "lip_aspect_ratio" in ts.columns
    assert meta["n_frames"] == 40
    assert meta["n_movements"] >= 1


def test_run_feature_computation_writes_feature_tables(tmp_path: Path):
    root = tmp_path
    norm_dir = root / "kinematics" / "004_normalization" / "tables"
    norm_dir.mkdir(parents=True)
    norm_csv = norm_dir / "video01-norm-lmks.csv"
    _norm_table().to_csv(norm_csv, index=False)
    pd.DataFrame([{"video_id": "video01", "output_csv": str(norm_csv), "status": "ok", "source_path": "video01.webm"}]).to_csv(norm_dir / "normalized_landmarks_manifest.csv", index=False)
    result = run_feature_computation(root, FeatureComputationConfig(use_smoothed_signals=False))
    assert Path(result["features_csv"]).exists()
    df = pd.read_csv(result["features_csv"])
    assert len(df) == 1
    assert "mouth_aperture_median" in df.columns
    assert Path(df.loc[0, "output_timeseries_csv"]).exists()
