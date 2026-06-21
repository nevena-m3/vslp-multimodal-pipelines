from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vslp.analysis.kinematics.features import FeatureComputationConfig, compute_feature_timeseries, feature_registry_dataframe
from vslp.core.feature_contract import FEATURE_REGISTRY_COLUMNS, normalize_feature_registry


def _point(index: int, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {f"{index}_x_norm": x, f"{index}_y_norm": y, f"{index}_z_norm": np.zeros_like(x)}


def test_kinematic_registry_has_cross_modal_formula_contract() -> None:
    registry = feature_registry_dataframe()
    assert len(registry) == 72
    assert set(FEATURE_REGISTRY_COLUMNS).issubset(registry.columns)
    assert registry["formula"].astype(str).str.len().gt(10).all()
    assert registry["source_document"].eq("Kinematic Feature Validation.xlsx").all()


def test_workbook_geometry_features_have_expected_values() -> None:
    t = np.linspace(0.0, 1.0, 31)
    data = {"timestamp_ms": t * 1000.0, "face_detected": True}
    data.update(_point(0, np.zeros_like(t), np.ones_like(t)))
    data.update(_point(17, np.zeros_like(t), -np.ones_like(t)))
    data.update(_point(61, -2 * np.ones_like(t), np.zeros_like(t)))
    data.update(_point(291, 2 * np.ones_like(t), np.zeros_like(t)))
    ts, _ = compute_feature_timeseries(pd.DataFrame(data), FeatureComputationConfig(use_smoothed_signals=False))
    assert np.nanmedian(ts["mouth_area_right"]) == pytest.approx(2.0)
    assert np.nanmedian(ts["mouth_area_left"]) == pytest.approx(2.0)
    assert np.nanmedian(ts["mouth_area_total"]) == pytest.approx(4.0)
    assert np.nanmedian(ts["mouth_area_abs_asymmetry"]) == pytest.approx(0.0)
    assert np.nanmedian(ts["lip_eccentricity"]) == pytest.approx(np.sqrt(0.75))


def test_acceleration_is_second_derivative_of_signed_position() -> None:
    t = np.linspace(0.0, 2.0, 121)
    opening = 0.25 * t**2 + 0.2
    data = {"timestamp_ms": t * 1000.0, "face_detected": True}
    data.update(_point(0, np.zeros_like(t), np.zeros_like(t)))
    data.update(_point(17, np.zeros_like(t), opening))
    ts, _ = compute_feature_timeseries(pd.DataFrame(data), FeatureComputationConfig(use_smoothed_signals=False))
    assert np.nanmedian(ts["mouth_aperture_acceleration"].to_numpy()[3:-3]) == pytest.approx(0.5, abs=0.02)
    assert np.nanmedian(np.abs(ts["mouth_aperture_jerk"].to_numpy()[5:-5])) < 0.03


def test_acoustic_registry_can_use_same_schema() -> None:
    source = pd.DataFrame([{"feature": "f0_mean", "subsystem": "phonatory", "meaning": "Mean F0"}])
    out = normalize_feature_registry(source, "acoustic")
    assert list(out.columns[: len(FEATURE_REGISTRY_COLUMNS)]) == list(FEATURE_REGISTRY_COLUMNS)
    assert out.loc[0, "family"] == "phonatory"
    assert out.loc[0, "modality"] == "acoustic"
