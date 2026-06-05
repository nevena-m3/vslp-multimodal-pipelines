from pathlib import Path
import pandas as pd

from vslp.analysis.features import AnalysisInputs, run_feature_analysis
from vslp.analysis.features.column_mapping import infer_column_roles, columns_by_role
from vslp.analysis.features.schemas import ColumnRole


def test_column_mapping_detects_features_and_identifiers():
    df = pd.DataFrame({
        "file_name": ["a.wav", "b.wav"],
        "subject_id": ["S1", "S2"],
        "task": ["bamboo", "bamboo"],
        "f0_mean": [100.0, 110.0],
        "percent_pause": [20.0, 30.0],
        "diagnosis": ["ALS", "control"],
    })
    mapping = infer_column_roles(df)
    assert "f0_mean" in columns_by_role(mapping, ColumnRole.FEATURE)
    assert "percent_pause" in columns_by_role(mapping, ColumnRole.FEATURE)
    assert "file_name" in columns_by_role(mapping, ColumnRole.IDENTIFIER)
    assert "task" in columns_by_role(mapping, ColumnRole.TASK)


def test_feature_analysis_writes_outputs(tmp_path: Path):
    features = pd.DataFrame({
        "file_name": ["a.wav", "b.wav", "c.wav", "d.wav"],
        "subject_id": ["S1", "S2", "S3", "S4"],
        "f0_mean": [100.0, 110.0, None, 130.0],
        "percent_pause": [20.0, 30.0, 80.0, 25.0],
    })
    qc = pd.DataFrame({"file_name": ["a.wav", "b.wav", "c.wav", "d.wav"], "snr_db": [30.0, 20.0, 5.0, 25.0]})
    fpath = tmp_path / "features.csv"; qpath = tmp_path / "qc.csv"
    features.to_csv(fpath, index=False); qc.to_csv(qpath, index=False)
    result = run_feature_analysis(AnalysisInputs(feature_table=fpath, qc_table=qpath, output_root=tmp_path))
    assert result.report_path.exists()
    assert result.tables["feature_distribution_summary"].exists()
    assert result.tables["feature_reliability_screen"].exists()
    assert result.plots["missingness_top_features"].exists()
