from pathlib import Path
import json
import pandas as pd

from vslp.analysis.kinematics import (
    artifact_inventory_dataframe,
    stage_status_dataframe,
    write_inspector_inventory,
    write_pipeline_summary_report,
)


def test_inspector_inventory_writes_stage_and_artifact_tables(tmp_path: Path):
    tables = tmp_path / "kinematics" / "000_ingest" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame([{"video_id": "v1", "fps": 30.0}]).to_csv(tables / "video_ingest_manifest.csv", index=False)

    paths = write_inspector_inventory(tmp_path)

    stage_csv = Path(paths["stage_status_csv"])
    artifact_csv = Path(paths["artifact_inventory_csv"])
    manifest_json = Path(paths["inspector_manifest_json"])
    assert stage_csv.exists()
    assert artifact_csv.exists()
    assert manifest_json.exists()

    stage_df = pd.read_csv(stage_csv)
    artifact_df = pd.read_csv(artifact_csv)
    ingest = stage_df.loc[stage_df["stage_id"] == "000_ingest"].iloc[0]
    assert ingest["status"] == "complete"
    assert "video_ingest_manifest.csv" in artifact_df["filename"].tolist()
    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "vslp_kinematics_inspector_v1"


def test_stage_status_marks_missing_required_outputs(tmp_path: Path):
    df = stage_status_dataframe(tmp_path)
    landmarks = df.loc[df["stage_id"] == "002_landmarks"].iloc[0]
    assert landmarks["status"] == "missing"
    assert landmarks["required_expected"] == 1


def test_pipeline_summary_report_uses_inspector_tables(tmp_path: Path):
    tables = tmp_path / "kinematics" / "007_aggregation" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame([{"video_id": "v1", "status": "ok", "mouth_aperture_median": 0.2}]).to_csv(
        tables / "kinematic_aggregated_features.csv", index=False
    )
    write_inspector_inventory(tmp_path)

    res = write_pipeline_summary_report(tmp_path)

    report = Path(res["report_html"])
    manifest = Path(res["manifest_json"])
    assert report.exists()
    assert manifest.exists()
    text = report.read_text(encoding="utf-8")
    assert "VSLP Kinematics Pipeline Summary" in text
    assert "kinematic_aggregated_features" in text
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema"] == "vslp_kinematics_report_manifest_v1"


def test_readiness_checklist_written_and_marks_visual_qc_as_known_gap(tmp_path: Path):
    from vslp.analysis.kinematics import readiness_checklist_dataframe

    qc_tables = tmp_path / "kinematics" / "005_video_qc" / "tables"
    qc_tables.mkdir(parents=True)
    (qc_tables / "video_qc_framework_placeholder.csv").write_text("family,status\nlighting,placeholder_to_be_built_in\n", encoding="utf-8")

    paths = write_inspector_inventory(tmp_path)

    readiness_csv = Path(paths["readiness_checklist_csv"])
    readiness_json = Path(paths["readiness_checklist_json"])
    assert readiness_csv.exists()
    assert readiness_json.exists()

    df = pd.read_csv(readiness_csv)
    assert "known_deferred_work" in set(df["category"])
    assert "known_gap" in set(df["status"])

    direct = readiness_checklist_dataframe(tmp_path)
    assert not direct.empty
    assert {"category", "check", "status", "recommended_action"}.issubset(direct.columns)
