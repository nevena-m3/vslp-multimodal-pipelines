from pathlib import Path

APP = Path('src/vslp/gui/features/app.py')
PLOTS = Path('src/vslp/analysis/features/plots.py')


def read(path):
    return path.read_text(encoding='utf-8')


def test_distribution_plot_names_are_professional_and_versioned():
    s = read(APP)
    assert 'APP_VERSION = "v0.108.0"' in (read(APP) if 'read' in globals() else APP.read_text(encoding='utf-8'))
    section = s[s.index('    def _distributions_page'):s.index('    def _refresh_dist_task_combo')]
    assert 'Shape and tail audit", "distribution_shape_story"' in section
    assert 'Range and outlier triage", "distribution_outlier_range_story"' in section
    assert 'Recording outlier burden", "row_outlier_story"' in section
    assert 'Shape story' not in section
    assert 'Recording outlier story' not in section


def test_distribution_captions_do_not_use_story_language():
    s = read(APP)
    caption_section = s[s.index('    def _distribution_plot_caption_text'):s.index('    def preview_distribution_plot')]
    assert 'Shape and tail audit:' in caption_section
    assert 'Range and outlier triage:' in caption_section
    assert 'Recording outlier burden:' in caption_section
    assert 'Shape story' not in caption_section
    assert 'Recording outlier story' not in caption_section


def test_distribution_plot_titles_do_not_use_story_language():
    s = read(PLOTS)
    assert 'Distribution shape and tail audit' in s
    assert 'Recording outlier burden' in s
    assert 'Distribution shape story' not in s
    assert 'Recording outlier burden story' not in s


def test_distribution_layout_is_more_responsive_on_smaller_screens():
    s = read(APP)
    dist_section = s[s.index('    def _distributions_page'):s.index('    def _refresh_dist_task_combo')]
    assert 'self.dist_plot_combo.setMinimumWidth(260)' in dist_section
    assert 'self.dist_task_combo.setMinimumWidth(200)' in dist_section
    assert 'self.dist_feature_combo.setMinimumWidth(240)' in dist_section
    assert 'side_panel.setMinimumWidth(260)' in dist_section
    assert 'side_panel.setMaximumWidth(340)' in dist_section
    assert 'side_panel.setFixedWidth(340)' not in dist_section
    assert 'self.dist_plot_preview.setMinimumHeight(440)' in dist_section
    assert 'widget.setMinimumWidth(860)' in s
