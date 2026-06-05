from vslp.acoustic.features.stage import FeatureExtractionConfig, _apply_computation_mode_defaults, _build_reduction_audit
from vslp.acoustic.features.registry import build_acoustic_feature_registry


def test_computation_mode_effective_task_sets_family_regions():
    cfg = FeatureExtractionConfig(computation_mode="effective_task_with_pauses")
    cfg = _apply_computation_mode_defaults(cfg)
    assert cfg.acoustic_region_policy == "effective_task"
    assert cfg.rhythm_region_policy == "effective_task"
    assert cfg.coordination_region_policy == "effective_task"
    assert cfg.phonatory_mode == "effective_task_exploratory"


def test_reduction_audit_documents_native_scale():
    registry = build_acoustic_feature_registry()
    registry = registry[registry["feature"].isin(["percent_pause", "CPP_mean", "f2", "ratio_below_above", "CPP_F1_comp"])]
    audit = _build_reduction_audit(registry, FeatureExtractionConfig())
    assert {"feature", "family", "native_measurement_scale", "file_level_scalar_reduction"}.issubset(audit.columns)
    assert audit.loc[audit["feature"].eq("percent_pause"), "family"].iloc[0] == "timing_respiratory"
    assert "voiced" in audit.loc[audit["feature"].eq("CPP_mean"), "native_measurement_scale"].iloc[0]
