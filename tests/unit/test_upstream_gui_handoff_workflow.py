from pathlib import Path

import json
import pandas as pd

from vslp.analysis.kinematics.metadata import link_metadata
from vslp.gui.features.app import FeatureAnalysisGUI


def _write_main_handoff(root: Path, modality_folder: str, manifest_modality: str) -> Path:
    main = root / modality_folder / "feature_handoff" / "main"
    main.mkdir(parents=True)
    pd.DataFrame({"recording_id": ["r1"], "task": ["bamboo"], "f1": [1.0]}).to_csv(
        main / "feature_values.csv", index=False
    )
    pd.DataFrame({"feature": ["f1"], "modality": [manifest_modality]}).to_csv(
        main / "feature_registry.csv", index=False
    )
    pd.DataFrame(columns=("recording_id", "feature", "status")).to_csv(
        main / "feature_status.csv", index=False
    )
    (main / "feature_export_manifest.json").write_text(
        json.dumps({"contract": "vslp-feature-handoff", "modality": manifest_modality}),
        encoding="utf-8",
    )
    return main


def test_feature_gui_resolves_acoustic_main_handoff(tmp_path: Path):
    main = _write_main_handoff(tmp_path, "acoustic", "acoustic")
    pd.DataFrame({"recording_id": ["r1"], "snr_db": [24.0]}).to_csv(main / "qc_features.csv", index=False)

    result = FeatureAnalysisGUI.resolve_feature_handoff_folder(main.parent)

    assert result["workspace"] == tmp_path
    assert result["modality"] == "Acoustic"
    assert result["feature"] == main / "feature_values.csv"
    assert result["qc"] == main / "qc_features.csv"
    assert result["metadata"] is None


def test_feature_gui_resolves_kinematics_modality_folder(tmp_path: Path):
    main = _write_main_handoff(tmp_path, "kinematics", "kinematic")
    result = FeatureAnalysisGUI.resolve_feature_handoff_folder(tmp_path / "kinematics")
    assert result["workspace"] == tmp_path
    assert result["modality"] == "Kinematic"
    assert result["main_dir"] == main


def test_kinematic_metadata_stage_accepts_no_metadata(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"video_id": ["v1"], "source_path": ["clip.mp4"]}).to_csv(manifest, index=False)

    result = link_metadata(manifest, None, tmp_path)

    assert result["link_mode"] == "no_metadata_provided"
    assert result["n_metadata_rows"] == 0
    assert pd.read_csv(result["link_preview"])["video_id"].tolist() == ["v1"]
    assert list(pd.read_csv(result["metadata_loaded"]).columns) == ["metadata_status"]


def test_kinematic_full_workflow_runs_real_stages_in_order():
    source = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    start = source.index("    def run_all(self)")
    end = source.index("    def _finish_full_workflow", start)
    block = source[start:end]
    ordered_calls = (
        "run_ingest(",
        "link_metadata(",
        "write_landmark_plan(",
        "run_mediapipe_landmarks(",
        "write_selected_landmarks(",
        "run_normalization_from_selection(",
        "run_video_qc(",
        "run_feature_computation(",
        "run_temporal_aggregation(",
        "write_inspector_inventory(",
        "write_pipeline_summary_report(",
    )
    positions = [block.index(call) for call in ordered_calls]
    assert positions == sorted(positions)
    assert block.count("self._start_worker(") == 1


def test_upstream_guis_expose_main_and_supplementary_outputs():
    acoustic = Path("src/vslp/gui/acoustic_app/main_window.py").read_text(encoding="utf-8")
    kinematic = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    features = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    for source in (acoustic, kinematic):
        assert "Open Main Feature GUI Handoff" in source
        assert "Open Supplementary Outputs" in source
    assert "Load Main Handoff Folder" in features
    assert "resolve_feature_handoff_folder" in features
