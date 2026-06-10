from vslp.analysis.kinematics import NORMALIZATION_METHOD_DETAILS, NORMALIZATION_METHODS


def test_normalization_methods_have_dashboard_metadata():
    assert set(NORMALIZATION_METHODS).issubset(set(NORMALIZATION_METHOD_DETAILS))
    for method in NORMALIZATION_METHODS:
        details = NORMALIZATION_METHOD_DETAILS[method]
        assert details["display_name"]
        assert details["what_changes"]
        assert details["best_for"]
        assert details["caution"]
        assert details["evidence_level"]
        assert isinstance(details["anchor_landmarks"], tuple)
        assert isinstance(details["fallback_landmarks"], tuple)


def test_intercanthal_method_declares_inner_canthus_and_outer_eye_fallback():
    details = NORMALIZATION_METHOD_DETAILS["intercanthal_distance"]
    assert details["anchor_landmarks"] == (133, 362)
    assert details["fallback_landmarks"] == (33, 263)
    assert "not" in details["caution"].lower()
