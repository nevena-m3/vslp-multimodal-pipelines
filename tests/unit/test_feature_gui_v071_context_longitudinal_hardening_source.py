from pathlib import Path


def test_feature_gui_v071_stable_plot_display_helper():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.72.0"' in text
    assert 'def _display_plot_image' in text
    assert 'label.clear()' in text
    assert 'preview.setMaximumHeight' in text
    assert 'def resizeEvent' in text
    assert 'self._display_plot_image(label, Path(path))' in text


def test_feature_gui_v071_longitudinal_feature_trajectory_controls():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'Selected feature trajectory' in text
    assert 'self.longitudinal_feature_combo' in text
    assert 'def _selected_longitudinal_feature' in text
    assert 'def _refresh_longitudinal_feature_combo' in text
    assert 'plot_longitudinal_feature_trajectory' in text
    assert 'long_feature_trajectory' in text


def test_feature_gui_v071_spearman_constant_warning_hardened():
    text = Path("src/vslp/analysis/features/audit.py").read_text(encoding="utf-8")
    assert 'def _safe_spearman' in text
    assert 'warnings.catch_warnings' in text
    assert 'input array is constant' in text
    assert 'dropna().nunique() > 1' in text
