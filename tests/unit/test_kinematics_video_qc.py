from pathlib import Path

import pandas as pd

from vslp.analysis.kinematics.video_qc import run_video_qc


def test_run_video_qc_writes_summary(tmp_path: Path):
    tables = tmp_path / "kinematics" / "002_landmarks" / "tables"
    tables.mkdir(parents=True)
    lmks = tables / "sample-lmks.csv"
    rows = []
    for i in range(6):
        detected = i not in {2, 3}
        row = {"frame": i, "timestamp_ms": i * 33, "face_detected": detected}
        for idx in [13, 14, 61, 291, 33, 263, 152, 1, 0, 17, 78, 308, 81, 311, 199]:
            row[f"{idx}_x"] = 0.4 + idx * 0.0001 + i * 0.001 if detected else None
            row[f"{idx}_y"] = 0.5 + idx * 0.0001 + i * 0.001 if detected else None
            row[f"{idx}_z"] = 0.0 if detected else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(lmks, index=False)
    manifest = tables / "landmarks_manifest.csv"
    pd.DataFrame([
        {"video_id": "sample", "source_path": "sample.webm", "output_csv": str(lmks), "status": "ok", "n_frames": 6, "n_faces_detected": 4, "face_detected_fraction": 4/6}
    ]).to_csv(manifest, index=False)

    res = run_video_qc(tmp_path, manifest)
    out = Path(res["summary_csv"])
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df["video_id"]) == ["sample"]
    assert "qc_status" in df.columns
    assert int(df.loc[0, "max_no_face_gap_frames"]) == 2


def test_kinematics_gui_exposes_video_qc_stage():
    app_py = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "Run Landmark / Video QC" in app_py
    assert "run_video_qc_stage" in app_py
    assert "_set_progress_busy" in app_py


def test_video_qc_reports_selected_validity_timestamps_and_normalization_flags(tmp_path: Path):
    tables = tmp_path / "kinematics" / "002_landmarks" / "tables"
    tables.mkdir(parents=True)
    lmks = tables / "qc-rich-lmks.csv"
    rows = []
    for i in range(5):
        detected = i != 2
        row = {"frame": i, "timestamp_ms": i * 40, "face_detected": detected}
        for idx in [13, 14]:
            row[f"{idx}_x"] = 0.5 + i * 0.001 if detected else None
            row[f"{idx}_y"] = 0.5 + i * 0.001 if detected else None
            row[f"{idx}_z"] = 0.0 if detected else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(lmks, index=False)
    manifest = tables / "landmarks_manifest.csv"
    pd.DataFrame([{"video_id": "qc-rich", "source_path": "sample.webm", "output_csv": str(lmks), "status": "ok"}]).to_csv(manifest, index=False)
    norm_dir = tmp_path / "kinematics" / "004_normalization" / "tables"
    norm_dir.mkdir(parents=True)
    pd.DataFrame([{
        "video_id": "qc-rich",
        "status": "qc_flagged",
        "qc_flags": "normalization_high_scale_cv",
        "scale_value_cv": 0.12,
        "scale_frame_to_frame_max_jump_fraction": 0.05,
    }]).to_csv(norm_dir / "normalized_landmarks_manifest.csv", index=False)

    res = run_video_qc(tmp_path, manifest)
    df = pd.read_csv(res["summary_csv"])

    assert (tmp_path / "kinematics" / "005_video_qc" / "tables" / "video_qc_summary.csv").exists()
    assert (tmp_path / "kinematics" / "005_video_qc" / "tables" / "video_qc_long_gaps.csv").exists()
    assert "selected_landmark_valid_rate" in df.columns
    assert "max_no_face_gap_s" in df.columns
    assert df.loc[0, "timestamp_monotonic"] in [True, "True", 1]
    assert df.loc[0, "normalization_status"] == "qc_flagged"
    assert "normalization" in str(df.loc[0, "qc_rationale"])
