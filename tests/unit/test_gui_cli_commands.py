from pathlib import Path

from typer.testing import CliRunner

from vslp.cli.main import app


runner = CliRunner()


def test_gui_help_lists_all_completed_desktop_applications():
    result = runner.invoke(app, ["gui", "--help"])

    assert result.exit_code == 0
    assert "acoustic" in result.stdout
    assert "kinematics" in result.stdout
    assert "features" in result.stdout


def test_pyproject_exposes_dedicated_gui_console_scripts():
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'vslp-acoustic-gui = "vslp.gui.acoustic_app.app:launch_acoustic_gui"' in text
    assert 'vslp-kinematics-gui = "vslp.gui.kinematics.app:launch_kinematics_gui"' in text
    assert 'vslp-features-gui = "vslp.gui.features.app:main"' in text
