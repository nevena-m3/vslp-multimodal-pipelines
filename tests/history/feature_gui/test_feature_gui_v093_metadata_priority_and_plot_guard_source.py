from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / 'src' / 'vslp' / 'gui' / 'features' / 'app.py'
AUDIT = ROOT / 'src' / 'vslp' / 'analysis' / 'features' / 'audit.py'


def text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def test_v093_version_and_metadata_priority_helpers_present():
    src = text(APP)
    assert 'APP_VERSION = "v0.93.0"' in src
    assert 'def _promote_metadata_context_columns' in src
    assert 'def _metadata_priority_context_fields' in src
    assert 'Metadata Mapping is the source of truth' in src
    assert 'Filename fallback remains active only in the no-metadata branch above' in src


def test_v093_metadata_branch_does_not_fill_canonical_context_from_filename_when_metadata_loaded():
    src = text(APP)
    start = src.index('# Metadata is present, so filename parsing is audit-only here')
    meta_branch = src[start : src.index('meta_df = self._standardize_metadata_table', start)]
    assert '_parse_filename_context_frame(feature_base)' in meta_branch
    assert '_fill_empty_context_from_filename(feature_base)' not in meta_branch
    promo_start = src.index('merged = self._promote_metadata_context_columns(merged)')
    post_merge = src[promo_start - 300 : promo_start + 300]
    assert '_promote_metadata_context_columns(merged)' in post_merge
    assert '_fill_empty_context_from_filename(merged)' not in post_merge


def test_v093_overview_plot_generation_logs_traceback_and_returns_partial_paths():
    src = text(APP)
    block = src[src.index('def _generate_overview_plots') : src.index('def preview_selected_overview_plot')]
    assert 'try:' in block
    assert 'except Exception as exc:' in block
    assert 'traceback.format_exc()' in block
    assert 'return paths' in block


def test_v093_pca_scores_defensively_aligns_scores_to_rows():
    src = text(AUDIT)
    block = src[src.index('def feature_pca_scores') : src.index('def feature_relationship_summary')]
    assert 'n = min(len(feature_df.index), int(scores.shape[0]))' in block
    assert 'scores[:n, i]' in block
    assert 'alignment_note' in block
