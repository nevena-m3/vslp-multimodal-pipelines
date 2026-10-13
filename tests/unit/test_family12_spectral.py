"""Legacy generic F12 IDs cannot be selected after task-specific migration."""

import pytest

from vslp.acoustic.features.catalog import load_feature_catalog
from vslp.acoustic.features.stage import FeatureExtractionConfig, _select_registry


@pytest.mark.parametrize("legacy_id", [
    "mfcc01_mean", "spectral_bandwidth_p2_mean_hz",
    "zcr_mean_fraction", "spec_m1_hz", "spectral_centroid_hz",
    "spectral_contrast_band0_db", "opensmile_1_4khz_<functional>",
])
def test_generic_f12_id_is_not_selectable(legacy_id):
    ids = {row["feature_id"] for row in load_feature_catalog()["outputs"]
           if row["family_id"] == "F12" and row["selectable"]}
    assert legacy_id not in ids
    with pytest.raises(ValueError, match="exact, approved output IDs"):
        _select_registry(FeatureExtractionConfig(selected_features=[legacy_id]))
