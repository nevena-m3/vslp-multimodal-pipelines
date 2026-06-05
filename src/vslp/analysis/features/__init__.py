"""Feature-analysis backend public API.

This module intentionally re-exports the stable objects used by tests, CLI,
and GUI code. Keep imports lightweight so importing ``vslp.analysis.features``
does not trigger GUI dependencies.
"""

from .core import AnalysisInputs, run_feature_analysis

try:  # Optional convenience export; older builds may not define this.
    from .core import AnalysisOutputs  # type: ignore
except ImportError:  # pragma: no cover
    AnalysisOutputs = None  # type: ignore

__all__ = ["AnalysisInputs", "AnalysisOutputs", "run_feature_analysis"]
