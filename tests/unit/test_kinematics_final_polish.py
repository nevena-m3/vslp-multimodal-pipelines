from pathlib import Path


def test_kinematics_gui_version_is_current():
    app_py = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "v0.87"' in app_py
    assert "Heavy MediaPipe extraction and final feature computation are connected" not in app_py


def test_kinematics_docs_do_not_use_windows_fragile_pytest_glob():
    offenders = []
    for path in Path("docs").glob("kinematics_gui_v0*.md"):
        text = path.read_text(encoding="utf-8")
        if "tests/unit/test_kinematics_*.py" in text:
            offenders.append(str(path))
    assert not offenders, "Fragile pytest glob found in: " + ", ".join(offenders)
