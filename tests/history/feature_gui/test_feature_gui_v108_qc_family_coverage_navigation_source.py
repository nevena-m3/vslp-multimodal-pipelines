from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_qc_version_v108():
    assert 'APP_VERSION = "v0.108.0"' in (read(APP) if 'read' in globals() else APP.read_text(encoding='utf-8'))


def test_automated_qc_speed_guard_preserves_six_acoustic_families():
    s = read(APP)
    func = s[s.index('    def _limit_qc_table_for_speed'):s.index('    def _qc_source_variable_table')]
    for family in [
        'Additive interference',
        'Gain / level dynamics',
        'Reverberation / echo',
        'Channel / device / platform',
        'Nonlinear distortion',
        'Temporal discontinuities',
    ]:
        assert family in func
    assert 'stratified by QC family' in func
    assert 'qc_family_from_name(c)' in func
    assert 'max_metrics = 60 if source_mode == "Manual QC" else 48' in func


def test_feature_mapping_continues_to_metadata_mapping_not_overview():
    s = read(APP)
    func = s[s.index('    def accept_mapping_and_continue'):s.index('    def update_mapping_summary')]
    assert 'continue to Metadata Mapping' in func
    assert 'self.show_page("metadata_mapping")' in func
    assert 'self.show_page("overview")' not in func


def test_deep_menus_prompt_for_feature_analysis_first():
    s = read(APP)
    assert 'def _analysis_required_pages' in s
    func = s[s.index('    def _analysis_required_pages'):s.index('    def _build_run_log_panel')]
    for key in ['missing', 'dist', 'qc', 'relationships', 'screening', 'reliability', 'ml_export']:
        assert f'"{key}"' in func
    assert 'Run Feature Analysis first' in func
    assert 'self.run_analysis()' in func


def test_analysis_ready_set_only_after_full_analysis():
    s = read(APP)
    func = s[s.index('    def run_analysis'):s.index('    def populate_output_tables')]
    assert 'self.analysis_ready = True' in func
    assert 'self.analysis_ready = False' in func
