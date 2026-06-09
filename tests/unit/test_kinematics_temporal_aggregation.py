from pathlib import Path

import numpy as np
import pandas as pd

from vslp.analysis.kinematics.aggregation import TemporalAggregationConfig, aggregate_timeseries_file, run_temporal_aggregation


def _make_feature_outputs(root: Path):
    ts_dir = root / "kinematics" / "006_features" / "timeseries"
    tab_dir = root / "kinematics" / "006_features" / "tables"
    ts_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.DataFrame({
        "frame": np.arange(10),
        "timestamp_ms": np.arange(10) * 33.3,
        "time_s": np.arange(10) * 0.0333,
        "face_detected": [True] * 9 + [False],
        "movement_id": [0] * 5 + [1] * 5,
        "movement_kind": ["open"] * 5 + ["close"] * 5,
        "mouth_aperture": np.linspace(0.1, 0.9, 10),
        "outer_lip_spread": np.linspace(1.0, 1.2, 10),
        "mouth_aperture_velocity": np.linspace(0.0, 0.2, 10),
        "mouth_aperture_raw": np.linspace(0.1, 0.9, 10),
    })
    ts_path = ts_dir / "vid01-kinematic-timeseries.csv"
    ts.to_csv(ts_path, index=False)
    pd.DataFrame([{"video_id": "vid01", "status": "ok", "output_timeseries_csv": str(ts_path), "feature_qc_flags": ""}]).to_csv(tab_dir / "kinematic_features.csv", index=False)
    return ts_path


def test_aggregate_timeseries_file_basic(tmp_path):
    ts_path = _make_feature_outputs(tmp_path)
    row, movement_rows = aggregate_timeseries_file(ts_path, TemporalAggregationConfig(profile="movement_segmented"), video_id="vid01")
    assert row["video_id"] == "vid01"
    assert row["status"] == "ok"
    assert row["n_movements"] == 2
    assert "mouth_aperture_median" in row
    assert len(movement_rows) == 2


def test_run_temporal_aggregation_writes_outputs(tmp_path):
    _make_feature_outputs(tmp_path)
    res = run_temporal_aggregation(tmp_path, TemporalAggregationConfig(profile="clinically_sensitive"))
    assert Path(res["aggregated_features_csv"]).exists()
    assert Path(res["movement_level_features_csv"]).exists()
    assert Path(res["manifest_json"]).exists()
    df = pd.read_csv(res["aggregated_features_csv"])
    assert list(df["video_id"]) == ["vid01"]
    assert "mouth_aperture_p95" in df.columns


def test_aggregation_gui_controls_present():
    app_py = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "Run Temporal Aggregation" in app_py
    assert "run_temporal_aggregation_stage" in app_py
    assert "kinematic_aggregated_features.csv" in app_py
