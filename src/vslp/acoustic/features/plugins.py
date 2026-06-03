"""Feature plugin interface.

Every future acoustic feature should be implemented as a plugin with explicit units,
parameters, failure handling, and references to its definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any
import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    subsystem: str
    unit: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


class AcousticFeaturePlugin(Protocol):
    spec: FeatureSpec

    def compute(self, x: np.ndarray, sr: int, context: dict[str, Any]) -> float:
        """Compute one scalar feature. Raise a clear exception on failure."""
        ...
