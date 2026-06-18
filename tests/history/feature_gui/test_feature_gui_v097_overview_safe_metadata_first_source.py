from pathlib import Path

APP = Path('src/vslp/gui/features/app.py').read_text(encoding='utf-8')


def test_version_bumped_to_v097():
    assert 'APP_VERSION = "v0.97.0"' in APP
    assert 'APP_VERSION = "v0.95.0"' not in APP


def test_overview_uses_emergency_metadata_first_builder():
    assert 'def _overview_emergency_outputs' in APP
    assert 'metadata_first' in APP
    assert 'Filename fallback remains active only in the no-metadata branch' in APP


def test_regenerate_overview_uses_guaranteed_plot_path():
    section = APP[APP.index('def regenerate_overview_plots'):APP.index('def preview_plot', APP.index('def regenerate_overview_plots'))]
    assert '_overview_emergency_outputs()' in section
    assert '_generate_overview_plots_guaranteed' in section
    assert 'Could not generate overview plots' in section


def test_guaranteed_plots_write_placeholders_on_failure():
    assert 'def _generate_overview_plots_guaranteed' in APP
    assert 'was replaced by placeholder' in APP
    assert 'def _write_overview_placeholder_plot' in APP
