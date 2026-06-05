from vslp.acoustic.features.registry import build_acoustic_feature_registry
from vslp.acoustic.features.scales import build_feature_computation_policy, build_feature_family_policy_summary


def test_feature_computation_policy_has_required_columns():
    reg = build_acoustic_feature_registry()
    policy = build_feature_computation_policy(reg)
    required = {
        "feature",
        "family_policy",
        "default_region",
        "native_measurements",
        "file_level_reduction",
        "avoid",
    }
    assert required.issubset(policy.columns)
    assert len(policy) == len(reg)


def test_key_feature_families_have_expected_policy():
    reg = build_acoustic_feature_registry()
    policy = build_feature_computation_policy(reg).set_index("feature")
    assert policy.loc["percent_pause", "family_policy"] == "Timing / respiratory"
    assert "pause" in policy.loc["percent_pause", "native_measurements"].lower()
    assert policy.loc["ratio_below_above", "default_region"] == "effective_task"
    assert "voiced" in policy.loc["f0_mean", "native_measurements"].lower()
    assert "lpc" in policy.loc["f2", "native_measurements"].lower()
    assert "spectral" in policy.loc["A1P0", "native_measurements"].lower()
    assert "trajector" in policy.loc["CPP_F1_comp", "native_measurements"].lower()


def test_family_policy_summary_is_gui_ready():
    summary = build_feature_family_policy_summary()
    assert set(summary["family"]) >= {
        "Timing / respiratory",
        "Rhythm / EMS",
        "Phonatory",
        "Articulatory / formant",
        "Resonatory / nasality",
        "Coordination",
    }
    assert summary["scalar_reduction"].notna().all()
