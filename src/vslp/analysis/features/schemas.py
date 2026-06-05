from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class ColumnRole(str, Enum):
    IDENTIFIER = "identifier"
    FEATURE = "feature"
    QC_FEATURE = "qc_feature"
    TARGET = "target"
    COVARIATE = "covariate"
    TASK = "task"
    TIME = "time"
    IGNORE = "ignore"


@dataclass
class AnalysisInputs:
    feature_table: Path
    output_root: Path
    qc_table: Optional[Path] = None
    metadata_table: Optional[Path] = None
    feature_registry: Optional[Path] = None
    modality: str = "auto"
    join_key: str = "auto"
    target_column: Optional[str] = None
    group_column: Optional[str] = None
    max_corr_features: int = 80


@dataclass
class AnalysisResult:
    output_root: Path
    report_path: Path
    tables: Dict[str, Path] = field(default_factory=dict)
    plots: Dict[str, Path] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
