import pandas as pd

from vslp.analysis.features.audit import reliability_subject_record_counts


def test_reliability_subject_record_counts_all_missing_subject_ids_returns_empty_schema():
    df = pd.DataFrame(
        {
            "subject_id": [None, None, None],
            "session_id": ["S1", "S1", "S2"],
            "task": ["bamboo", "pa", "bamboo"],
            "f0_mean": [110.0, 112.0, 108.0],
        }
    )

    out = reliability_subject_record_counts(df)

    assert list(out.columns) == ["subject", "n_records", "n_sessions", "n_tasks", "interpretation"]
    assert out.empty
