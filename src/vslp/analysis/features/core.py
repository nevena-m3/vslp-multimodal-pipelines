"""Backward-compatible public API for feature analysis."""
from .schemas import AnalysisInputs, AnalysisResult, ColumnRole
from .pipeline import run_feature_analysis

__all__ = ["AnalysisInputs", "AnalysisResult", "ColumnRole", "run_feature_analysis"]
