from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"
PLOTS = ROOT / "src" / "vslp" / "analysis" / "features" / "plots.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_qc_version_v107():
    assert 'APP_VERSION = "v0.108.0"' in (read(APP) if 'read' in globals() else APP.read_text(encoding='utf-8'))


def test_manual_qc_role_mapping_excludes_non_manual_roles():
    s = read(APP)
    func = s[s.index('    def _manual_qc_family_for_role'):s.index('    def _manual_qc_family_allowed')]
    assert 'if role and not role.startswith("--"):' in func
    assert 'return None' in func
    assert '"speech"' not in func
    assert '"audio"' not in func.split('# Legacy/header fallback only when no assigned metadata role is known.')[-1]


def test_manual_qc_columns_use_metadata_mapping_as_source_of_truth():
    s = read(APP)
    func = s[s.index('    def _manual_qc_columns_from_mapping'):s.index('    def _manual_qc_dataframe')]
    assert 'assigned Metadata Mapping roles as the source of' in func
    assert 'if rows:' in func and 'return rows' in func
    assert 'inferred_from_header_no_mapping' in func
    assert 'inferred_from_header")' not in func


def test_manual_qc_output_columns_are_family_prefixed():
    s = read(APP)
    func = s[s.index('    def _manual_qc_dataframe'):s.index('    def _automated_qc_dataframe')]
    assert 'manual_acquisition_qc' in func
    assert 'manual_audio_qc' in func
    assert 'manual_video_qc' in func
    assert 'manual_face_visibility_qc' in func
    assert 'metric = f"{family_prefix.get(family' in func


def test_manual_qc_distribution_is_flag_prevalence_not_blank_histogram_grid():
    s = read(PLOTS)
    func = s[s.index('def plot_qc_metric_distributions'):s.index('def plot_qc_top_feature_associations')]
    assert 'is_manual' in func
    assert 'Manual QC flag prevalence' in func
    assert 'Flagged fraction of recordings' in func
    assert 'all missing' not in func
