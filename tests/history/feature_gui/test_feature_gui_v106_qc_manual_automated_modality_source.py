from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"
AUDIT = ROOT / "src" / "vslp" / "analysis" / "features" / "audit.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_qc_version_and_binary_source_options():
    s = read(APP)
    assert 'APP_VERSION = "v0.108.0"' in (read(APP) if 'read' in globals() else APP.read_text(encoding='utf-8'))
    assert 'self.qc_source_combo.addItems(["Manual QC", "Automated QC"])' in s
    assert "Manual + automated" not in s
    assert "Manual + automated QC" not in s
    assert "self.qc_framework_combo" not in s


def test_manual_qc_modality_groups_are_explicit():
    s = read(APP)
    assert 'if modality == "acoustic":' in s
    assert 'return family in {"Manual acquisition QC", "Manual audio QC"}' in s
    assert 'if modality == "kinematic":' in s
    assert '"Manual video QC", "Manual face/visibility QC"' in s
    assert 'def _manual_qc_dataframe' in s
    assert 'def _automated_qc_dataframe' in s


def test_automated_qc_table_only_used_for_automated_source():
    s = read(APP)
    scope = s[s.index('    def _qc_scope_tables'):s.index('    def _refresh_qc_task_combo')]
    assert 'if source == "Manual QC":' in scope
    assert 'q = self._manual_qc_dataframe(f)' in scope
    assert 'else:' in scope
    assert 'q = self._automated_qc_dataframe(row_mask)' in scope


def test_manual_metadata_qc_aliases_and_families():
    s = read(APP)
    assert '"task_completed_as_instructed": "Manual acquisition QC flag"' in s
    assert '"needs_parsing": "Manual acquisition QC flag"' in s
    assert '"wearing_glasses": "Manual face/visibility QC flag"' in s
    assert '"facial_hair_present": "Manual face/visibility QC flag"' in s
    a = read(AUDIT)
    assert 'return "Manual acquisition QC"' in a
    assert 'return "Manual audio QC"' in a
    assert 'return "Manual video QC"' in a
    assert 'return "Manual face/visibility QC"' in a


def test_qc_speed_caps_present():
    s = read(APP)
    assert 'def _limit_qc_table_for_speed' in s
    assert 'max_metrics = 60 if source_mode == "Manual QC" else 48' in s
    assert 'max_features = 70' in s
