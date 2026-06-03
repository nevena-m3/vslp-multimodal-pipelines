"""Local pretrained-model registry scaffold.

Pretrained models must be local files. No cloud inference or external model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import joblib


@dataclass(frozen=True)
class ModelCard:
    model_id: str
    model_type: str
    target_label: str
    feature_table_schema: str
    trained_at_utc: str
    group_split_column: str
    notes: str = "Research use only."

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def save_sklearn_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_sklearn_model(path: str | Path):
    return joblib.load(path)
