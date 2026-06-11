from pathlib import Path


def test_feature_gui_v072_imports_warnings_for_analysis_context():
    text = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.72.0"' in text
    assert 'import warnings' in text
    assert 'warnings.catch_warnings()' in text


def test_feature_gui_v072_safe_spearman_imports_warnings_locally():
    text = Path("src/vslp/analysis/features/audit.py").read_text(encoding="utf-8")
    assert 'def _safe_spearman' in text
    assert 'import warnings as _warnings' in text
    assert '_warnings.catch_warnings()' in text
    assert 'return float("nan")' in text
