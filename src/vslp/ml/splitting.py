"""Leakage-safe ML splitting utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def subject_level_train_test_split(df: pd.DataFrame, subject_col: str = "subject_id", test_size: float = 0.2, random_state: int = 42):
    """Split rows while ensuring all sessions/iterations from a subject stay together."""
    if subject_col not in df.columns:
        raise ValueError(f"Missing required subject/group column: {subject_col}")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[subject_col]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
