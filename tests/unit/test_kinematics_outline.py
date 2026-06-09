from pathlib import Path

from vslp.analysis.kinematics.schemas import WORKFLOW_STAGES, STAGE_GUIDANCE
from vslp.analysis.kinematics.reports import write_scaffold_report


def test_workflow_outline_has_expected_stages():
    assert len(WORKFLOW_STAGES) == 10
    assert WORKFLOW_STAGES[0][0] == "Setup"
    assert WORKFLOW_STAGES[-1][0] == "Reports"
    for key in ["setup", "metadata", "landmarks", "selection", "normalization", "qc", "features", "aggregation", "inspector", "reports"]:
        assert key in STAGE_GUIDANCE
        assert STAGE_GUIDANCE[key]["purpose"]
        assert STAGE_GUIDANCE[key]["decision"]


def test_branding_assets_packaged():
    root = Path(__file__).parents[2]
    assert (root / "src" / "vslp" / "gui" / "kinematics" / "assets" / "branding" / "lab_logo.png").exists()
    assert (root / "src" / "vslp" / "gui" / "kinematics" / "assets" / "branding" / "uoft_logo.png").exists()


def test_scaffold_report_contains_commercial_sections(tmp_path: Path):
    path = write_scaffold_report(tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "VSLP Kinematics GUI Workflow Outline" in html
    assert "Scientific guardrails" in html
    assert "Temporal aggregation profiles" in html
