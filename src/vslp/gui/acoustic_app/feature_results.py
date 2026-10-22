"""Human-readable summaries of the per-recording feature status table."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd


def result_summary(path: str | Path, *, include_rows: bool = True) -> tuple[dict, pd.DataFrame]:
    """Classify every selected recording/leaf result without hiding unavailable rows."""
    if not include_rows:
        counts: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        recordings: set[str] = set()
        features: set[str] = set()
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"recording_id", "feature_id", "status", "failure_reason", "value"}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError("feature_status_schema_missing")
            for row in reader:
                recordings.add(row["recording_id"])
                features.add(row["feature_id"])
                status = row["status"].casefold()
                reason = row["failure_reason"]
                severity = {"computed": "COMPUTED", "warning": "WARNING",
                            "error": "ERROR", "unavailable": "UNAVAILABLE"}.get(status, "ERROR")
                if reason.startswith(("family_executor_failed", "executor_missing_result")):
                    severity = "ERROR"
                counts[severity] += 1
                if severity == "UNAVAILABLE":
                    reasons[reason] += 1
        return ({"selected_features": len(features), "recordings": len(recordings),
                 "results": sum(counts.values()), "computed": counts["COMPUTED"],
                 "unavailable": counts["UNAVAILABLE"], "warning": counts["WARNING"],
                 "error": counts["ERROR"], "reasons": dict(reasons)}, pd.DataFrame())
    rows = pd.read_csv(path, keep_default_na=False)
    required = {"recording_id", "feature_id", "status", "failure_reason", "value"}
    if not required <= set(rows):
        raise ValueError("feature_status_schema_missing")
    if rows.duplicated(["recording_id", "feature_id"]).any():
        raise ValueError("duplicate_feature_status_row")
    reasons = rows.failure_reason.astype(str)
    status = rows.status.astype(str).str.casefold()
    severity = status.map({"computed": "COMPUTED", "warning": "WARNING",
                           "error": "ERROR", "unavailable": "UNAVAILABLE"})
    runtime_failure = reasons.str.startswith(("family_executor_failed", "executor_missing_result"))
    severity = severity.mask(runtime_failure, "ERROR").fillna("ERROR")
    rows = rows.copy()
    rows.insert(rows.columns.get_loc("status") + 1, "severity", severity)
    counts = severity.value_counts()
    return ({"selected_features": int(rows.feature_id.nunique()),
             "recordings": int(rows.recording_id.nunique()),
             "results": len(rows),
             "computed": int(counts.get("COMPUTED", 0)),
             "unavailable": int(counts.get("UNAVAILABLE", 0)),
             "warning": int(counts.get("WARNING", 0)),
             "error": int(counts.get("ERROR", 0)),
             "reasons": rows.loc[severity.eq("UNAVAILABLE"), "failure_reason"]
                            .value_counts().to_dict()}, rows)
