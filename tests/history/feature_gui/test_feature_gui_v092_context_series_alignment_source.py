from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')


def test_context_alignment_helper_exists_for_scoped_menu_dataframes():
    assert 'APP_VERSION = "v0.92.0"' in APP
    assert 'def _align_context_series_to_frame(self, series: pd.Series | None, frame: pd.DataFrame)' in APP
    assert 'series.reindex(frame.index)' in APP
    assert 'pd.Series(pd.NA, index=frame.index, dtype="string")' in APP


def test_missingness_scope_assigns_aligned_context_not_raw_values():
    start = APP.index('def _missingness_scope_table')
    end = APP.index('def _missingness_feature_cols')
    block = APP[start:end]
    assert 'aligned_series = self._align_context_series_to_frame(series, scoped)' in block
    assert 'scoped["__local_missingness_context__"] = aligned_series' in block
    assert 'scoped["__local_missingness_context__"] = series.values' not in block


def test_distribution_scope_assigns_aligned_context_not_raw_values():
    start = APP.index('def _distribution_scope_table')
    end = APP.index('def _distribution_feature_cols')
    block = APP[start:end]
    assert 'aligned_series = self._align_context_series_to_frame(series, out)' in block
    assert 'out["__local_clinical_context__"] = aligned_series' in block
    assert 'out["__local_clinical_context__"] = series.values' not in block


def test_task_review_assigns_aligned_clinical_context_not_raw_values():
    start = APP.index('def generate_task_review_plots')
    end = APP.index('def preview_task_review_plot')
    block = APP[start:end]
    assert 'aligned_series = self._align_context_series_to_frame(clinical_series, df)' in block
    assert 'df[label_col] = aligned_series' in block
    assert 'df[label_col] = clinical_series.values' not in block
