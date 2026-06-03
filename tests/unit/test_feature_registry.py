from vslp.acoustic.features.registry import build_acoustic_feature_registry


def test_feature_registry_has_73_features():
    df = build_acoustic_feature_registry()
    assert len(df) == 73
    assert df["feature"].is_unique
    assert set(["feature", "subsystem", "meaning", "unit", "computation_note"]).issubset(df.columns)
