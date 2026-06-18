from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"
PLOTS = ROOT / "src" / "vslp" / "analysis" / "features" / "plots.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_version_and_source_selector_present():
    text = read(APP)
    assert 'APP_VERSION = "v0.128.0"' in text
    assert "Reliability source:" in text
    assert "Acoustic features" in text
    assert "Automated QC metrics" in text
    assert "def _selected_reliability_source" in text


def test_reliability_task_scope_preserved_for_both_sources():
    text = read(APP)
    assert "Task focus:" in text
    assert "same-task reliability scope" in text
    assert "same-task repeated units" in text or "repeated same-task" in text
    assert "_align_automated_qc_to_analysis" in text
    assert "Automated QC reliability is meaningful when QC rows can be aligned" in text


def test_feature_family_classification_no_blind_unclassified_fallback():
    text = read(APP)
    assert "def _infer_acoustic_family_from_feature_name" in text
    assert "Other mapped numeric feature" in text
    assert "normalize_name(f)" in text
    assert "family_lookup.get(feat, qc_family_from_name(feat)" in text


def test_automated_qc_reliability_uses_qc_families():
    text = read(APP)
    assert "qc_family_from_name(str(c))" in text
    assert "self._longitudinal_qc_numeric_columns(df)" in text
    assert "reliability_source" in text


def test_reliability_plots_have_interpretability_layers():
    text = read(PLOTS)
    assert "Interpretation bands" in text
    assert "Interpretation layer" in text
    assert "Stable: ICC >= .75" in text
    assert "MDC95 / total SD" in text
    assert "Family-level screen" in text
