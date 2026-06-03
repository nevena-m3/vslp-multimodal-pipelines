"""Feature plugin interfaces for VSLP acoustic feature extraction.

Each plugin owns a small, testable group of features. Plugins should be pure
computational units: they receive file-level context and return feature values plus
per-feature status notes. The stage orchestrates file I/O, manifests, reports, and
plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class FeatureContext:
    """Per-file inputs made available to feature plugins."""

    file_name: str
    row: pd.Series
    segments_csv: Path | None = None
    segmentation_wav_path: Path | None = None
    task: str | None = None
    duration_sec: float | None = None
    config: Any | None = None


@dataclass
class FeatureValue:
    """One feature value and its audit status."""

    feature: str
    value: float | int | str | None
    status: str = "computed"
    note: str = ""


class AcousticFeaturePlugin(Protocol):
    """Protocol implemented by acoustic feature plugins."""

    subsystem: str
    feature_names: tuple[str, ...]

    def compute(self, context: FeatureContext) -> dict[str, FeatureValue]:
        """Compute feature values for one file."""


def as_feature_values(features: dict[str, float], note: str, status: str = "computed") -> dict[str, FeatureValue]:
    return {name: FeatureValue(feature=name, value=value, status=status, note=note) for name, value in features.items()}
