from pathlib import Path

import pandas as pd

from vslp.analysis.kinematics.features import (
    feature_framework_dataframe,
    feature_implementation_audit_dataframe,
    write_feature_framework_catalog,
)


def test_feature_framework_separates_implemented_kernel_from_roadmap():
    framework = feature_framework_dataframe()
    audit = feature_implementation_audit_dataframe()
    assert not framework.empty
    assert {"Feature family", "Current implementation", "Evidence status"}.issubset(framework.columns)
    assert framework["Current implementation"].str.contains("implemented kernel|partial", case=False, regex=True).any()
    assert not audit.empty
    assert audit["Audit item"].str.contains("Landmark convention").any()


def test_write_feature_framework_catalog_outputs_tables(tmp_path: Path):
    paths = write_feature_framework_catalog(tmp_path)
    assert paths["framework_csv"].exists()
    assert paths["audit_csv"].exists()
    assert paths["framework_json"].exists()
    df = pd.read_csv(paths["framework_csv"])
    assert "Vertical lip/jaw opening" in set(df["Feature family"])
