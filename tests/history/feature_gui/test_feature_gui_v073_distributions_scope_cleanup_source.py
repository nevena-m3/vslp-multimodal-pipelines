from pathlib import Path


def test_feature_gui_v073_distribution_removes_redundant_group_overlay():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.86.0"' in text
    assert 'Group overlay:' not in text
    assert 'self.dist_group_combo' not in text
    assert 'Grouping for selected-feature plots comes from Clinical context above.' in text
    assert 'There is no separate group overlay control.' in text


def test_feature_gui_v073_distribution_uses_clinical_context_for_grouping():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'def _selected_distribution_group' in text
    assert 'single grouping source' in text
    assert '"__local_clinical_context__"' in text
    assert 'fallback' in text.lower()
    assert 'self.preview_distribution_plot(self.dist_plot_combo.currentData()' in text
