from pathlib import Path

import pandas as pd

from vslp.analysis.kinematics.aggregation import (
    TemporalAggregationConfig,
    aggregation_guide_dataframe,
    aggregate_timeseries_file,
    run_temporal_aggregation,
    write_aggregation_guide,
)


def test_aggregation_guide_explains_statistics_and_settings(tmp_path: Path):
    df = aggregation_guide_dataframe()
    assert {"section", "item", "meaning", "output_effect", "recommended_use"}.issubset(df.columns)
    assert "median" in set(df["item"])
    assert "minimum valid fraction" in set(df["item"])
    out = write_aggregation_guide(tmp_path)
    assert Path(out["aggregation_guide_csv"]).exists()
    assert Path(out["aggregation_guide_json"]).exists()


def test_temporal_aggregation_writes_guide_and_preserves_timeseries_policy(tmp_path: Path):
    root = tmp_path
    features_dir = root / "kinematics" / "006_features" / "tables"
    features_dir.mkdir(parents=True)
    ts_path = features_dir / "v1-kinematic-timeseries.csv"
    pd.DataFrame({
        "frame": [0, 1, 2, 3],
        "time_s": [0.0, 0.1, 0.2, 0.3],
        "face_detected": [True, True, True, True],
        "mouth_aperture": [0.1, 0.2, 0.3, 0.4],
        "mouth_aperture_velocity": [1.0, 1.2, 1.4, 1.6],
    }).to_csv(ts_path, index=False)
    features_csv = features_dir / "kinematic_features.csv"
    pd.DataFrame({
        "video_id": ["v1"],
        "output_timeseries_csv": [str(ts_path)],
        "status": ["ok"],
        "feature_qc_flags": [""],
    }).to_csv(features_csv, index=False)

    res = run_temporal_aggregation(root, TemporalAggregationConfig())
    assert Path(res["aggregated_features_csv"]).exists()
    assert Path(res["aggregation_guide_csv"]).exists()
    agg = pd.read_csv(res["aggregated_features_csv"])
    assert "mouth_aperture_median" in agg.columns
    assert "mouth_aperture_valid_fraction" in agg.columns
    assert agg.loc[0, "mouth_aperture_median"] == 0.25


def test_aggregation_flags_low_valid_fraction():
    ts = pd.DataFrame({
        "frame": [0, 1, 2, 3],
        "time_s": [0.0, 0.1, 0.2, 0.3],
        "face_detected": [True, True, True, True],
        "mouth_aperture": [0.1, float("nan"), float("nan"), 0.4],
    })
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v2-kinematic-timeseries.csv"
        ts.to_csv(path, index=False)
        row, _ = aggregate_timeseries_file(path, TemporalAggregationConfig(min_valid_fraction=0.75), video_id="v2")
    assert "low_valid_fraction:mouth_aperture" in row["aggregation_qc_flags"]
    assert row["status"] == "qc_flagged"
