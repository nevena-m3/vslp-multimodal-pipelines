from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vslp.analysis.kinematics.normalization import (
    NormalizationConfig,
    run_normalization,
    run_normalization_from_selection,
)


def _make_landmark_csv(path: Path) -> None:
    rows = []
    for frame in range(4):
        row = {"frame": frame, "timestamp_ms": frame * 33, "face_detected": True}
        for idx in range(478):
            row[f"{idx}_x"] = 0.5 + idx * 0.0001
            row[f"{idx}_y"] = 0.4 + idx * 0.0001
            row[f"{idx}_z"] = 0.01
        # Stable anchors for intercanthal distance.
        row["133_x"], row["133_y"], row["133_z"] = 0.40, 0.40, 0.0
        row["362_x"], row["362_y"], row["362_z"] = 0.60, 0.40, 0.0
        row["1_x"], row["1_y"], row["1_z"] = 0.50, 0.50, 0.0
        row["13_x"], row["13_y"], row["13_z"] = 0.50, 0.55 + frame * 0.01, 0.0
        row["14_x"], row["14_y"], row["14_z"] = 0.50, 0.60 + frame * 0.01, 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_normalization_from_selection_writes_manifest_and_normalized_csv(tmp_path: Path) -> None:
    tables = tmp_path / "kinematics" / "002_landmarks" / "tables"
    tables.mkdir(parents=True)
    lmks = tables / "vid1-lmks.csv"
    _make_landmark_csv(lmks)
    pd.DataFrame([{"video_id": "vid1", "source_path": "video.webm", "output_csv": str(lmks), "status": "ok"}]).to_csv(tables / "landmarks_manifest.csv", index=False)
    sel_dir = tmp_path / "kinematics" / "003_selection" / "tables"
    sel_dir.mkdir(parents=True)
    (sel_dir / "selected_landmarks.json").write_text(json.dumps({"selected_landmarks": [13, 14], "preset": "unit_test"}), encoding="utf-8")

    result = run_normalization_from_selection(tmp_path, "intercanthal_distance")

    assert result["n_videos"] == 1
    manifest = Path(result["manifest_csv"])
    assert manifest.exists()
    out_df = pd.read_csv(manifest)
    assert out_df.loc[0, "status"] == "ok"
    normalized_csv = Path(out_df.loc[0, "output_csv"])
    assert normalized_csv.exists()
    norm = pd.read_csv(normalized_csv)
    assert "13_y_norm" in norm.columns
    assert "14_y_norm" in norm.columns
    assert abs(float(norm.loc[0, "scale_value_video_median"]) - 0.2) < 1e-9


def test_raw_normalized_coordinates_method_uses_unit_scale(tmp_path: Path) -> None:
    tables = tmp_path / "kinematics" / "002_landmarks" / "tables"
    tables.mkdir(parents=True)
    lmks = tables / "vid2-lmks.csv"
    _make_landmark_csv(lmks)
    pd.DataFrame([{"video_id": "vid2", "source_path": "video.webm", "output_csv": str(lmks), "status": "ok"}]).to_csv(tables / "landmarks_manifest.csv", index=False)

    cfg = NormalizationConfig(method="raw_normalized_coordinates", selected_landmarks=(13,), selected_preset="unit")
    result = run_normalization(tmp_path, cfg)

    out_df = pd.read_csv(result["manifest_csv"])
    norm = pd.read_csv(out_df.loc[0, "output_csv"])
    assert set(norm["scale_source"]) == {"raw_unit_scale"}
    assert set(norm["scale_value_video_median"]) == {1.0}
