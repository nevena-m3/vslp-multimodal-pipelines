from pathlib import Path


def test_preprocess_gui_is_minimal_and_dc_is_user_selectable():
    source = Path("src/vslp/gui/acoustic_app/main_window.py").read_text(encoding="utf-8")

    assert 'QCheckBox("Remove DC offset")' in source
    assert "remove_dc_offset=bool(self.remove_dc_check.isChecked())" in source
    assert '"Fixed scientific policy"' not in source
    assert '"Per-recording audit"' not in source
    assert "preview_latest_preprocess_plot" not in source
    assert "open_preprocess_plot_folder" not in source
    assert '"Show filters"' not in source
    assert '"Peak-normalize waveform"' not in source
