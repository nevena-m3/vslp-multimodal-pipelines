from pathlib import Path

SRC = Path('src/vslp/gui/features/app.py')


def text():
    return SRC.read_text(encoding='utf-8')


def test_overview_plot_generation_is_overview_only():
    src = text()
    block = src.split('def _generate_overview_plots', 1)[1].split('def preview_selected_overview_plot', 1)[0]
    assert 'Generate only Overview plots' in block
    assert 'missingness_top_features' not in block
    assert 'relationship_pca_scores' not in block
    assert 'screening_effect_ranking' not in block
    assert 'recommendation_counts' not in block


def test_overview_has_lightweight_output_builder():
    src = text()
    assert 'def _build_overview_outputs' in src
    block = src.split('def _build_overview_outputs', 1)[1].split('def regenerate_overview_plots', 1)[0]
    assert 'feature_distribution_summary' in block
    assert 'overview_feature_quality_landscape' in block
    assert 'feature_pca_scores' not in block
    assert 'build_group_outcome_screening' not in block


def test_regenerate_overview_uses_lightweight_builder():
    src = text()
    block = src.split('def regenerate_overview_plots', 1)[1].split('def preview_plot', 1)[0]
    assert ('self._build_overview_outputs()' in block) or ('self._overview_emergency_outputs()' in block)
    assert 'self._safe_refresh_analysis_dashboards(outputs)' not in block
    assert 'self.update_overview_dashboard(outputs)' in block


def test_run_analysis_falls_back_to_overview_only():
    src = text()
    block = src.split('def run_analysis', 1)[1].split('def populate_output_tables', 1)[0]
    assert 'Full feature analysis tables failed; falling back to Overview-only outputs' in block
    assert 'self._build_overview_outputs()' in block
    assert 'Overview-only analysis complete' in block
