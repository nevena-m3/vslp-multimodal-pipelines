from pathlib import Path


def test_feature_gui_v060_overview_source_layout_contract():
    text = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "v0.65.0"' in text
    assert 'Dataset snapshot' in text
    assert 'Detailed overview tables' in text
    assert 'Open current plot' in text
    assert 'Role mapping summary' in text
    block = text[text.index('def _overview_page'):text.index('def _metric_tile')]
    assert 'Top missing features' not in block
    assert 'Feature availability heatmap' not in block


def test_feature_gui_v060_overview_captions_are_nonredundant():
    text = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')
    block = text[text.index('def _overview_page'):text.index('def _metric_tile')]
    for label in [
        'Readiness scorecard',
        'Design context',
        'Role mapping summary',
        'Feature-family coverage',
        'Feature-quality landscape',
        'Subject x task coverage',
    ]:
        assert label in block
    assert 'Missingness, distribution, QC, and ML-export plots are handled in their own menus' in block
