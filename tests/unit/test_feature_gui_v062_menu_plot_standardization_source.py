from pathlib import Path


def test_feature_gui_v062_version_and_shared_plot_gallery():
    text = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "v0.65.0"' in text
    assert 'def _add_standard_plot_gallery' in text
    assert 'Open current plot' in text
    assert 'Detailed tables' in text


def test_feature_gui_v062_core_menus_use_standard_gallery():
    text = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    for attr in [
        'missing_plot_combo',
        'dist_plot_combo',
        'qc_plot_combo',
        'relationship_plot_combo',
        'screening_plot_combo',
        'reliability_plot_combo',
        'recommendation_plot_combo',
    ]:
        assert attr in text
    for phrase in [
        'Missingness uses only availability plots',
        'Distribution review focuses on shape',
        'QC Integration keeps only acquisition-sensitivity plots',
        'Feature Relationships keeps redundancy',
        'Group/Outcome Screening keeps balance',
        'Reliability keeps repeated-measure support',
        'Feature Recommendation keeps only integrated readiness plots',
    ]:
        assert phrase in text
