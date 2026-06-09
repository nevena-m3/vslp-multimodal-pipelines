from pathlib import Path


def test_feature_stage_uses_acoustic_style_selector():
    app_py = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "Feature selector" in app_py
    assert "QTreeWidget" in app_py
    assert "Default oral-motor" in app_py
    assert "QC requirements for selected features" in app_py
    assert "Run Kinematic Feature Computation" in app_py


def test_kinematic_feature_registry_exports():
    from vslp.analysis.kinematics import (
        DEFAULT_KINEMATIC_FEATURE_IDS,
        KINEMATIC_FEATURE_GROUPS,
        KINEMATIC_FEATURE_SPECS,
        QC_FEATURE_REQUIREMENTS,
        feature_registry_dataframe,
    )

    assert len(KINEMATIC_FEATURE_SPECS) >= 8
    assert DEFAULT_KINEMATIC_FEATURE_IDS
    assert KINEMATIC_FEATURE_GROUPS
    assert QC_FEATURE_REQUIREMENTS
    df = feature_registry_dataframe()
    assert {"feature_id", "group", "landmarks", "normalization"}.issubset(df.columns)
