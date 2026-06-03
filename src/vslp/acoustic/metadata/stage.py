"""Acoustic metadata stage.

This stage creates a clean file-level index used by feature extraction and later ML.
It supports two sources of metadata:
1. user-provided demographics/metadata CSV joined by file_name;
2. conservative filename parsing fallback.

The stage never silently invents clinical labels. Generated/inferred fields are flagged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import re

import pandas as pd

from vslp.acoustic.ingest.stage import run_acoustic_ingest
from vslp.core.project import ensure_stage_folders
from vslp.core.provenance import python_environment
from vslp.core.schemas import ArtifactRef, StageManifest, StageResult

REQUIRED_METADATA_COLUMNS = [
    "subject_id",
    "session_id",
    "iteration",
    "task",
    "recording_date",
    "diagnosis",
    "severity_score",
    "severity_bin",
    "file_name",
]

TASK_ALIASES = {
    "BAMBOO": "bamboo",
    "BAMBOOPASSAGE": "bamboo",
    "DDK": "ddk",
    "PA": "ddk_pa",
    "TA": "ddk_ta",
    "KA": "ddk_ka",
    "PATAKA": "ddk_pataka",
    "VOWEL": "sustained_vowel",
    "VOWELA": "sustained_vowel_a",
    "VOWEL_A": "sustained_vowel_a",
    "VC3": "vc3",
    "VC3G": "vc3g",
    "BOBBY": "buy_bobby_a_puppy",
    "BUYBOBBYAPUPPY": "buy_bobby_a_puppy",
    "SENTENCE": "sentence_reading",
    "SENTENCES": "sentence_reading",
    "READING": "sentence_reading",
}

@dataclass(frozen=True)
class MetadataConfig:
    """Metadata harmonization configuration."""

    demographics_csv: str | None = None
    allow_filename_parsing: bool = True
    filename_subject_pattern: str = r"(?i)(SUBJ|S|ID)[_-]?(?P<subject>[A-Za-z0-9]+)"
    filename_session_pattern: str = r"(?i)(SESSION|SESS)[_-]?(?P<session>[A-Za-z0-9]+)"
    filename_iteration_pattern: str = r"(?i)(ITERATION|ITER|VISIT|V)[_-]?(?P<iteration>[0-9]+)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if key in {"filename", "file", "audio_file", "audio_filename"}:
            key = "file_name"
        elif key in {"id", "participant_id", "subject", "subject"}:
            key = "subject_id"
        elif key in {"session", "visit", "visit_id"}:
            key = "session_id"
        elif key in {"date", "recordingdate", "recording_date"}:
            key = "recording_date"
        elif key in {"dx", "group", "diagnostic_group"}:
            key = "diagnosis"
        elif key in {"severity", "alsfrs", "alsfrs_r", "score"}:
            key = "severity_score"
        rename[col] = key
    return df.rename(columns=rename)


def _parse_task_from_name(name: str) -> tuple[str | None, str | None]:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", Path(name).stem).upper().strip("_")
    tokens = [t for t in cleaned.split("_") if t]
    joined = "".join(tokens)
    for token in tokens:
        if token in TASK_ALIASES:
            return TASK_ALIASES[token], f"task parsed from token '{token}'"
    for alias, task in TASK_ALIASES.items():
        if alias in joined:
            return task, f"task parsed from filename alias '{alias}'"
    return None, None


def _parse_filename_metadata(file_name: str, cfg: MetadataConfig, file_index: int) -> dict[str, Any]:
    stem = Path(file_name).stem
    out: dict[str, Any] = {}
    notes: list[str] = []

    subject = None
    m = re.search(cfg.filename_subject_pattern, stem)
    if m:
        subject = m.groupdict().get("subject")
    if not subject:
        # Conservative fallback: first token when the filename starts with a clear ID-like prefix.
        first = re.split(r"[_\-\s]+", stem)[0]
        if re.match(r"(?i)^(SUBJ|S|ID)?[A-Za-z]*\d+", first):
            subject = first
            notes.append("subject_id inferred from first filename token")
    if not subject:
        subject = f"AUTO_SUBJECT_{file_index:04d}"
        notes.append("subject_id generated because filename parsing failed")
    out["subject_id"] = str(subject)

    session = None
    m = re.search(cfg.filename_session_pattern, stem)
    if m:
        session = m.groupdict().get("session")
    if not session:
        session = "SESSION01"
        notes.append("session_id defaulted to SESSION01")
    out["session_id"] = str(session)

    iteration = None
    m = re.search(cfg.filename_iteration_pattern, stem)
    if m:
        iteration = m.groupdict().get("iteration")
    if not iteration:
        iteration = "1"
        notes.append("iteration defaulted to 1")
    out["iteration"] = str(iteration)

    task, task_note = _parse_task_from_name(file_name)
    out["task"] = task if task else "unknown_task"
    if task_note:
        notes.append(task_note)
    else:
        notes.append("task could not be parsed; set to unknown_task")

    out["recording_date"] = pd.NA
    out["diagnosis"] = pd.NA
    out["severity_score"] = pd.NA
    out["severity_bin"] = pd.NA
    out["metadata_source"] = "filename_parser"
    out["metadata_notes"] = "; ".join(notes)
    return out


def _load_demographics(path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    if "file_name" not in df.columns:
        raise ValueError("Demographics CSV must contain a file_name column for v0.6 metadata joining.")
    for col in REQUIRED_METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
            warnings.append(f"demographics CSV missing optional/expected column '{col}'; column filled with NA")
    # Preserve only the first row for duplicate filenames; report duplicates.
    dup = df["file_name"].duplicated().sum()
    if dup:
        warnings.append(f"demographics CSV contains {dup} duplicate file_name rows; first occurrence retained")
        df = df.drop_duplicates("file_name", keep="first")
    return df, warnings


def run_acoustic_metadata(
    input_path: str | Path,
    output_root: str | Path,
    config: MetadataConfig | None = None,
) -> StageResult:
    """Create a clean project file index with metadata and audit flags."""
    cfg = config or MetadataConfig()
    output_root = Path(output_root)
    stage_dir = output_root / "acoustic" / "000_metadata"
    folders = ensure_stage_folders(stage_dir)

    ingest_result = run_acoustic_ingest(input_path=input_path, output_root=output_root)
    ingest_df = pd.read_csv(ingest_result.summary_table) if ingest_result.summary_table else pd.DataFrame()
    if ingest_df.empty:
        raise ValueError("No ingestable files found; cannot create metadata index.")

    rows: list[dict[str, Any]] = []
    for i, row in ingest_df.reset_index(drop=True).iterrows():
        file_name = row.get("file_name")
        parsed = _parse_filename_metadata(str(file_name), cfg=cfg, file_index=i + 1) if cfg.allow_filename_parsing else {}
        base = row.to_dict()
        base.update(parsed)
        base["metadata_status"] = "parsed_from_filename"
        rows.append(base)
    index_df = pd.DataFrame(rows)

    warnings: list[str] = []
    if cfg.demographics_csv:
        demo_path = Path(cfg.demographics_csv).expanduser()
        demo_df, demo_warnings = _load_demographics(demo_path)
        warnings.extend(demo_warnings)
        join_cols = [c for c in REQUIRED_METADATA_COLUMNS if c in demo_df.columns]
        demo_small = demo_df[join_cols].copy()
        merged = index_df.merge(demo_small, on="file_name", how="left", suffixes=("_parsed", "_csv"))
        for col in [c for c in REQUIRED_METADATA_COLUMNS if c != "file_name"]:
            parsed_col = f"{col}_parsed"
            csv_col = f"{col}_csv"
            if csv_col in merged.columns:
                merged[col] = merged[csv_col].combine_first(merged[parsed_col] if parsed_col in merged.columns else pd.NA)
            elif parsed_col in merged.columns:
                merged[col] = merged[parsed_col]
        csv_hit = merged[[f"{c}_csv" for c in REQUIRED_METADATA_COLUMNS if c != "file_name" and f"{c}_csv" in merged.columns]].notna().any(axis=1)
        merged["metadata_status"] = csv_hit.map(lambda x: "joined_from_csv" if x else "filename_parser_fallback")
        drop_cols = [c for c in merged.columns if c.endswith("_parsed") or c.endswith("_csv")]
        index_df = merged.drop(columns=drop_cols)

    for col in REQUIRED_METADATA_COLUMNS:
        if col not in index_df.columns:
            index_df[col] = pd.NA

    # Deterministic subject/session/task key for downstream grouping.
    index_df["record_key"] = (
        index_df["subject_id"].astype(str) + "__" +
        index_df["session_id"].astype(str) + "__ITER" +
        index_df["iteration"].astype(str) + "__" +
        index_df["task"].astype(str)
    )

    completeness_rows = []
    for col in REQUIRED_METADATA_COLUMNS:
        completeness_rows.append({
            "column": col,
            "fraction_present": float(index_df[col].notna().mean()),
            "n_missing": int(index_df[col].isna().sum()),
            "n_total": int(len(index_df)),
        })
    completeness_df = pd.DataFrame(completeness_rows)

    index_path = folders["tables"] / "project_file_index.csv"
    completeness_path = folders["tables"] / "metadata_completeness.csv"
    errors_path = folders["errors"] / "metadata_errors.csv"
    report_path = folders["reports"] / "metadata_report.html"

    index_df.to_csv(index_path, index=False)
    completeness_df.to_csv(completeness_path, index=False)
    pd.DataFrame([]).to_csv(errors_path, index=False)
    _write_metadata_report(report_path, index_df, completeness_df, warnings, cfg)

    manifest = StageManifest(
        stage_name="acoustic_metadata",
        stage_version="0.1.0",
        status="completed_with_warnings" if warnings else "completed",
        output_artifacts=[
            ArtifactRef(path=str(index_path), role="project_file_index", media_type="text/csv"),
            ArtifactRef(path=str(completeness_path), role="metadata_completeness", media_type="text/csv"),
            ArtifactRef(path=str(report_path), role="metadata_html_report", media_type="text/html"),
        ],
        config=cfg.to_dict(),
        environment={"python": python_environment()},
        warnings=warnings,
        notes=["Clinical labels are never invented; missing labels remain NA."],
    )
    manifest_path = folders["logs"] / "stage_manifest.json"
    manifest.write_json(manifest_path)
    return StageResult(
        status=manifest.status,
        manifest_path=manifest_path,
        summary_table=index_path,
        error_table=errors_path,
        report_path=report_path,
    )


def _write_metadata_report(path: Path, index_df: pd.DataFrame, completeness_df: pd.DataFrame, warnings: list[str], cfg: MetadataConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completeness_html = completeness_df.to_html(index=False, escape=True)
    status_counts = index_df["metadata_status"].value_counts(dropna=False).to_frame("count").reset_index().rename(columns={"index": "metadata_status"})
    status_html = status_counts.to_html(index=False, escape=True)
    warning_html = "".join(f"<li>{w}</li>" for w in warnings) if warnings else "<li>No warnings.</li>"
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>VSLP Metadata Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; background:#071A2D; color:#EAF2F8; margin:32px; }}
.card {{ background:#0B253D; border:1px solid #315D7C; border-radius:12px; padding:18px; margin:14px 0; }}
table {{ border-collapse: collapse; width: 100%; background:#102A43; }}
th, td {{ border:1px solid #315D7C; padding:7px; text-align:left; }}
th {{ background:#0E3A5B; }}
code {{ color:#8FCBFF; }}
</style></head><body>
<h1>VSLP Acoustic Metadata Report</h1>
<div class='card'><b>Files indexed:</b> {len(index_df)}<br><b>Demographics CSV:</b> {cfg.demographics_csv}</div>
<div class='card'><h2>Metadata source status</h2>{status_html}</div>
<div class='card'><h2>Completeness</h2>{completeness_html}</div>
<div class='card'><h2>Warnings</h2><ul>{warning_html}</ul></div>
<div class='card'><p>Missing clinical labels are kept as NA. Filename-derived values are explicitly flagged.</p></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
