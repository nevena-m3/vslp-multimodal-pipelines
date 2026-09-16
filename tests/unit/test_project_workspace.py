import json
from pathlib import Path

import pytest

from vslp.core.project import WORKSPACE_DIRS, initialize_project, register_project_component


def test_shared_workspace_has_one_root_and_isolated_components(tmp_path: Path):
    paths = initialize_project(tmp_path, project_name="ALS Study")

    assert paths.root == tmp_path.resolve()
    assert set(WORKSPACE_DIRS.values()) == {
        "configs",
        "acoustic",
        "kinematics",
        "feature_analysis",
        "ml",
        "logs",
    }
    assert all((paths.root / directory).is_dir() for directory in WORKSPACE_DIRS.values())
    assert not (paths.root / "kinematic").exists()


def test_component_registration_preserves_workspace_identity(tmp_path: Path):
    initialize_project(tmp_path, project_name="ALS Study")
    first = json.loads((tmp_path / "project_manifest.json").read_text(encoding="utf-8"))

    register_project_component(tmp_path, "acoustic", {"gui_version": "v1"}, project_name="Different label")
    register_project_component(tmp_path, "kinematics", {"gui_version": "v2"})
    manifest = json.loads((tmp_path / "project_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "vslp_workspace"
    assert manifest["project_name"] == "ALS Study"
    assert manifest["created_at_utc"] == first["created_at_utc"]
    assert set(manifest["components"]) == {"acoustic", "kinematics"}
    assert manifest["components"]["acoustic"]["directory"] == "acoustic"
    assert manifest["components"]["kinematics"]["directory"] == "kinematics"


def test_invalid_existing_manifest_is_not_silently_replaced(tmp_path: Path):
    (tmp_path / "project_manifest.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="not readable"):
        initialize_project(tmp_path)


def test_guis_use_current_workspace_contracts_and_modality_scoped_feature_outputs():
    acoustic = Path("src/vslp/gui/acoustic_app/main_window.py").read_text(encoding="utf-8")
    kinematics = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    features = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")

    # Acoustic GUI now creates an immutable task-specific acoustic run rather than
    # registering itself into a cross-modality workspace. Dedicated run-layout
    # tests validate the exact acoustic-run schema and directory contract.
    assert "initialize_acoustic_run" in acoustic
    assert 'component="acoustic"' not in acoustic

    # Kinematics and modality-neutral Feature Analysis retain their shared-workspace
    # contracts until those applications are migrated separately.
    assert 'component="kinematics"' in kinematics
    assert '/ "feature_analysis" / scope' in features
    assert '"Select modality (required)"' in features
