"""Nonclinical numerical and registry checks for Group 2 harmonic difference."""

import numpy as np
import pytest

from vslp.acoustic.features.catalog import load_feature_catalog, task_fit
from vslp.acoustic.features.family02_harmonic_difference import (
    corrected_h1_h2_db, iseli_formant_boost_db, mannell_bandwidth_hz,
    raw_h1_h2_db,
)


def test_raw_h1_h2_is_signed_db_difference():
    assert raw_h1_h2_db(4, 4) == 0
    assert raw_h1_h2_db(7, 4) == 3
    assert raw_h1_h2_db(1, 4) == -3


def test_iseli_equation_uses_both_formants_and_bandwidths():
    fs = 48000
    formants = (500., 1500.)
    bandwidths = (92., 116.)
    omega = 2 * np.pi * 200 / fs
    expected = 0.
    for f, b in zip(formants, bandwidths, strict=True):
        r = np.exp(-np.pi * b / fs)
        theta = 2 * np.pi * f / fs
        expected += 10 * np.log10(
            (1 - 2 * r * np.cos(theta) + r**2)**2 /
            ((1 - 2 * r * np.cos(omega + theta) + r**2) *
             (1 - 2 * r * np.cos(omega - theta) + r**2)))
    actual = iseli_formant_boost_db(200, fs, formants, bandwidths)
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(iseli_formant_boost_db(
        200, fs, (700., 1500.), bandwidths))
    assert actual != pytest.approx(iseli_formant_boost_db(
        200, fs, formants, (150., 116.)))
    assert actual != pytest.approx(iseli_formant_boost_db(
        200, fs, formants, (92., 200.)))
    assert mannell_bandwidth_hz(500) == pytest.approx(92)
    with pytest.raises(ValueError, match="invalid_formant_correction_inputs"):
        iseli_formant_boost_db(200, fs, formants, (0., 116.))


def test_corrected_differs_from_raw_and_recovers_synthetic_source_difference():
    fs = 48000
    f0 = 200.
    formants = (400., 1500.)
    bandwidths = (75., 116.)
    source_h1, source_h2 = 10., 7.
    observed_h1 = source_h1 + iseli_formant_boost_db(f0, fs, formants, bandwidths)
    observed_h2 = source_h2 + iseli_formant_boost_db(2 * f0, fs, formants, bandwidths)
    assert raw_h1_h2_db(observed_h1, observed_h2) != pytest.approx(3.)
    assert corrected_h1_h2_db(observed_h1, observed_h2, f0, fs,
                              formants, bandwidths) == pytest.approx(3.)
    assert np.isfinite(corrected_h1_h2_db(4, 2, 200, fs, formants, bandwidths))


def test_project_construct_and_task_recommendations():
    catalog = load_feature_catalog()
    construct = next(c for c in catalog["constructs"] if c["construct_id"] == "C080")
    outputs = {o["feature_id"]: o for o in construct["outputs"]}
    assert set(outputs) == {"h1_h2_db_raw", "h1star_h2star_db"}
    assert all(o["selectable"] for o in outputs.values())
    assert construct["id_provenance"] == "PROJECT_DEFINED_ID"
    corrected = outputs["h1star_h2star_db"]
    raw = outputs["h1_h2_db_raw"]
    assert task_fit(corrected, "Sustained /a/") == "Recommended"
    assert task_fit(raw, "Sustained /a/") == "Not specified"
    for task in ("S5", "S6", "Bamboo Passage", "DDK"):
        assert task_fit(corrected, task) == "Not specified"
        assert corrected["executable_for_task"]
