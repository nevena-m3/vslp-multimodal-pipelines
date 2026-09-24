"""Task identity and user-confirmed type are independent Setup fields."""

import json
import os

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from vslp.acoustic.run_setup import TASK_TYPES, initialize_acoustic_run
from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    return AcousticPipelineWindow()


def test_task_type_requires_explicit_gui_selection_and_never_follows_name():
    window = _window()
    assert window.task_type_combo.currentData() is None
    window.task_name_edit.setText("WSTG")
    assert window.task_type_combo.currentData() is None
    window.task_type_combo.setCurrentIndex(
        window.task_type_combo.findData("Sentence / controlled speech"))
    assert window.task_type_combo.currentData() == "Sentence / controlled speech"
    window.task_name_edit.setText("Bamboo Passage")
    assert window.task_type_combo.currentData() == "Sentence / controlled speech"
    assert tuple(window.task_type_combo.itemText(i) for i in range(1, 6)) == TASK_TYPES
    window.close()


def test_manifest_persists_exact_task_identity_and_selected_type(tmp_path):
    incoming = tmp_path / "input"
    incoming.mkdir()
    run = initialize_acoustic_run(
        project_name="P", task_name="WSTG", task_type="Sentence / controlled speech",
        input_folder=incoming, output_parent=tmp_path / "out",
        setup_values={"project_name": "P", "task_name": "WSTG",
                      "task_type": "Sentence / controlled speech",
                      "input_folder": str(incoming), "output_parent": str(tmp_path / "out")})
    manifest = json.loads((run.root / "project_manifest.json").read_text(encoding="utf-8"))
    setup = json.loads((run.root / "configs" / "setup_config.json").read_text(encoding="utf-8"))
    assert (manifest["task_id"], manifest["task_display_name"], manifest["task_type"]) == (
        "wstg", "WSTG", "Sentence / controlled speech")
    assert setup["task_type"] == "Sentence / controlled speech"
    with pytest.raises(ValueError, match="explicit Setup choices"):
        initialize_acoustic_run(
            project_name="P", task_name="WSTG", task_type="guessed sentence",
            input_folder=incoming, output_parent=tmp_path / "out",
            setup_values={})


def test_legacy_run_without_type_displays_not_specified(tmp_path, monkeypatch):
    incoming = tmp_path / "input"
    incoming.mkdir()
    run = initialize_acoustic_run(
        project_name="P", task_name="WSTG", input_folder=incoming,
        output_parent=tmp_path / "out", setup_values={})
    manifest_path = run.root / "project_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("task_type")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    window = _window()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(run.root))
    window.open_existing_run()
    assert window.task_type_combo.currentData() is None
    assert window.task_type_combo.currentText() == "Not specified"
    assert not window.task_type_combo.isEnabled()
    assert "Not specified" in window.feature_task_label.text()
    window.close()
