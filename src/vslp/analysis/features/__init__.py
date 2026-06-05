"""Feature Analysis utilities for VSLP.

This package is modality-neutral. It can inspect VSLP acoustic outputs,
future VSLP kinematic outputs, or generic user-uploaded feature tables.
"""

from .schemas import ColumnRole, AnalysisInputs, AnalysisResult
from .pipeline import run_feature_analysis

__all__ = ["ColumnRole", "AnalysisInputs", "AnalysisResult", "run_feature_analysis"]
