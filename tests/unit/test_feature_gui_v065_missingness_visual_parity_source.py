from pathlib import Path


def test_feature_gui_v065_missingness_snapshot_stack_matches_overview_pattern():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.67.0"' in text
    block = text[text.index('def update_missingness_dashboard'):text.index('def _missingness_plot_caption_text')]
    assert 'self.missing_metric_grid.addWidget(self._metric_tile(title, value, subtitle), idx, 0)' in block
    assert '("Features", n_features, "selected predictors")' in block
    assert '("Rows", len(row) if row is not None else "-", "recordings / files")' in block
    assert '("Mean missingness", mean_miss, "across selected predictors")' in block
    assert '("High-missing features", high_features, ">=50% missing or worse")' in block
    assert '("Metadata groups", groups, "available strata")' in block
    assert '("Default policy", "do not auto-impute", "review mechanism first")' in block
