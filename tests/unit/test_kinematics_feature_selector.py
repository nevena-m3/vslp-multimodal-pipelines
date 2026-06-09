from pathlib import Path

from vslp.analysis.kinematics.features import (
    KINEMATIC_FEATURE_GROUPS,
    KINEMATIC_FEATURE_SPECS,
    DEFAULT_KINEMATIC_FEATURE_IDS,
    QC_FEATURE_REQUIREMENTS,
)


def test_kinematic_feature_registry_contains_uploaded_feature_groups():
    groups = set(KINEMATIC_FEATURE_GROUPS)
    assert "Vertical lip/jaw displacement" in groups
    assert "Horizontal lip spread" in groups
    assert "Lip aperture geometry" in groups
    assert "Jaw lateralization" in groups
    assert "Lip symmetry" in groups
    assert "Bilateral coordination" in groups
    ids = {spec.feature_id for spec in KINEMATIC_FEATURE_SPECS}
    assert {"sLL_vert", "aLL_horz", "lip_aspect", "jaw_lateralization", "lip_symmetry", "lat_xcorr"}.issubset(ids)
    assert DEFAULT_KINEMATIC_FEATURE_IDS


def test_feature_registry_records_normalization_and_landmark_dependencies():
    for spec in KINEMATIC_FEATURE_SPECS:
        assert spec.landmarks
        assert "intercanthal" in spec.normalization.lower()
        assert spec.aggregation
    assert any(row["Parameter"] == "ICD availability" for row in QC_FEATURE_REQUIREMENTS)


def test_kinematics_gui_feature_selector_controls_present():
    app_py = Path("src/vslp/gui/kinematics/app.py").read_text(encoding="utf-8")
    assert "Feature selector" in app_py
    assert "Write Feature Computation Plan" in app_py
    assert "QC requirements for selected features" in app_py
    assert "QTreeWidget" in app_py
