from pathlib import Path


def test_feature_gui_v060_overview_source_layout_contract():
    text = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "v0.61.0"' in text
    assert 'Dataset snapshot' in text
    assert 'Detailed overview tables' in text
    assert 'Open current plot' in text
    assert 'Role mapping summary' in text
    block = text[text.index('def _overview_page'):text.index('def _metric_tile')]
    assert 'Top missing features' not in block
    assert 'Feature availability heatmap' not in block


def test_feature_gui_v060_overview_captions_are_nonredundant():
    text = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    block = text[text.index('def _overview_plot_caption_text'):text.index('def regenerate_overview_plots')]
    for key in [
        'overview_readiness_scorecard',
        'overview_design_tiles',
        'role_counts',
        'overview_feature_family_quality',
        'overview_feature_quality_landscape',
        'overview_subject_task_matrix',
    ]:
        assert key in block
    assert 'detailed missingness' in block.lower() or 'detailed diagnosis' in block.lower()
