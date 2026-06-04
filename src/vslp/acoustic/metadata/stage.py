"""Acoustic metadata stage.

This stage creates a clean file-level index used by feature extraction and later ML.
It supports flexible metadata CSVs whose column names vary across projects. The
stage detects likely canonical columns, links only uploaded/ingested files to the
metadata table, and writes audit tables for mappings, matches, unmatched files,
and unused metadata rows.

The stage never silently invents clinical labels. Generated/inferred fields are
flagged explicitly.
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

OPTIONAL_CANONICAL_COLUMNS = [
    "protocol_id",
    "clinical_visit_id",
    "assessment_date",
    "sex",
    "date_of_birth",
    "alsfrs_total",
    "alsfrs_bulbar",
    "alsbdi_total",
    "sentence_intelligibility_percent",
    "speaking_rate",
    "task_completed_as_instructed",
    "background_noise",
    "poor_audio_quality",
]

TASK_ALIASES = {
    "BAMBOO": "bamboo",
    "BAMBOOPASSAGE": "bamboo",
    "PASSAGE": "bamboo",
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

EXTENSION_PRIORITY = {".wav": 0, ".mp4": 1, ".webm": 2}

COLUMN_RULES: dict[str, list[tuple[str, int]]] = {
    "file_name": [
        (r"(^|[_\s-])(raw[_\s-]*)?media[_\s-]*file[_\s-]*name($|[_\s-])", 120),
        (r"(^|[_\s-])file[_\s-]*name($|[_\s-])", 110),
        (r"(^|[_\s-])filename($|[_\s-])", 110),
        (r"(^|[_\s-])(audio|video|media)[_\s-]*(file|filename|name)($|[_\s-])", 95),
    ],
    "extension": [(r"(^|[_\s-])extension($|[_\s-])", 100), (r"(^|[_\s-])ext($|[_\s-])", 80)],
    "subject_id": [
        (r"(^|[_\s-])subject[_\s-]*id($|[_\s-])", 120),
        (r"(^|[_\s-])subject($|[_\s-])", 100),
        (r"(^|[_\s-])participant[_\s-]*id($|[_\s-])", 100),
        (r"(^|[_\s-])patient[_\s-]*id($|[_\s-])", 90),
        (r"(^|[_\s-])id[_\s-]*norm($|[_\s-])", 85),
        (r"(^|[_\s-])id($|[_\s-])", 70),
    ],
    "protocol_id": [(r"(^|[_\s-])protocol[_\s-]*id($|[_\s-])", 120), (r"(^|[_\s-])protocol($|[_\s-])", 90)],
    "session_id": [
        (r"(^|[_\s-])session[_\s-]*id($|[_\s-])", 120),
        (r"(^|[_\s-])clinical[_\s-]*visit[_\s-]*id($|[_\s-])", 115),
        (r"(^|[_\s-])visit[_\s-]*id($|[_\s-])", 110),
        (r"(^|[_\s-])session($|[_\s-])", 95),
    ],
    "iteration": [
        (r"(^|[_\s-])iteration($|[_\s-])", 120),
        (r"(^|[_\s-])visit[_\s-]*(number|num|no|index)?($|[_\s-])", 95),
        (r"(^|[_\s-])longitudinal[_\s-]*(visit|iteration)($|[_\s-])", 105),
    ],
    "task": [
        (r"(^|[_\s-])task[_\s-]*name($|[_\s-])", 120),
        (r"(^|[_\s-])task($|[_\s-])", 110),
        (r"(^|[_\s-])speech[_\s-]*task($|[_\s-])", 100),
    ],
    "recording_date": [
        (r"(^|[_\s-])recording[_\s-]*date($|[_\s-])", 120),
        (r"(^|[_\s-])visit[_\s-]*date($|[_\s-])", 115),
        (r"(^|[_\s-])collection[_\s-]*date($|[_\s-])", 100),
        (r"(^|[_\s-])date($|[_\s-])", 80),
    ],
    "assessment_date": [(r"(^|[_\s-])assessment[_\s-]*date($|[_\s-])", 120), (r"(^|[_\s-])clinical[_\s-]*date($|[_\s-])", 90)],
    "diagnosis": [(r"(^|[_\s-])diagnosis($|[_\s-])", 120), (r"(^|[_\s-])dx($|[_\s-])", 90), (r"(^|[_\s-])group($|[_\s-])", 60)],
    "severity_score": [
        (r"(^|[_\s-])severity[_\s-]*score($|[_\s-])", 120),
        (r"(^|[_\s-])alsfrs[_\s-]*(r[_\s-]*)?total[_\s-]*score($|[_\s-])", 115),
        (r"(^|[_\s-])alsfrs[_\s-]*total($|[_\s-])", 110),
        (r"(^|[_\s-])alsfrs($|[_\s-])", 80),
    ],
    "severity_bin": [(r"(^|[_\s-])severity[_\s-]*bin($|[_\s-])", 120), (r"(^|[_\s-])severity[_\s-]*class($|[_\s-])", 100)],
    "alsfrs_total": [(r"(^|[_\s-])alsfrs[_\s-]*(r[_\s-]*)?total[_\s-]*score($|[_\s-])", 120), (r"(^|[_\s-])alsfrs[_\s-]*total($|[_\s-])", 115)],
    "alsfrs_bulbar": [(r"(^|[_\s-])alsfrs[_\s-]*bulbar[_\s-]*(subscore|score)?($|[_\s-])", 120)],
    "alsbdi_total": [(r"(^|[_\s-])alsbdi[_\s-]*total[_\s-]*score($|[_\s-])", 120), (r"(^|[_\s-])alsbdi[_\s-]*total($|[_\s-])", 115)],
    "sex": [(r"(^|[_\s-])sex($|[_\s-])", 120), (r"(^|[_\s-])gender($|[_\s-])", 80)],
    "date_of_birth": [(r"(^|[_\s-])date[_\s-]*of[_\s-]*birth($|[_\s-])", 120), (r"(^|[_\s-])dob($|[_\s-])", 100)],
    "sentence_intelligibility_percent": [(r"sentence[_\s-]*intelligibility[_\s-]*percent", 120), (r"intelligibility", 70)],
    "speaking_rate": [(r"(^|[_\s-])speaking[_\s-]*rate($|[_\s-])", 120), (r"(^|[_\s-])speech[_\s-]*rate($|[_\s-])", 80)],
    "task_completed_as_instructed": [(r"task[_\s-]*completed[_\s-]*as[_\s-]*instructed", 120)],
    "background_noise": [(r"background[_\s-]*noise", 120)],
    "poor_audio_quality": [(r"poor[_\s-]*audio[_\s-]*quality", 120)],
}

@dataclass(frozen=True)
class MetadataConfig:
    """Metadata harmonization configuration."""

    demographics_csv: str | None = None
    allow_filename_parsing: bool = True
    filename_subject_pattern: str = r"(?i)(SUBJ|S|ID)[_-]?(?P<subject>[A-Za-z0-9]+)"
    filename_session_pattern: str = r"(?i)(SESSION|SESS)[_-]?(?P<session>[A-Za-z0-9]+)"
    filename_iteration_pattern: str = r"(?i)(ITERATION|ITER|VISIT|V)[_-]?(?P<iteration>[0-9]+)"
    metadata_stem_match: bool = True
    preserve_all_metadata_columns: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _column_key(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def _score_column(canonical: str, col: str) -> tuple[int, str]:
    key = _column_key(col)
    best_score = 0
    best_rule = ""

    # Prevent common false positives, e.g. mapping "Date of Diagnosis" to diagnosis.
    if canonical == "diagnosis" and "date" in key:
        return 0, "excluded: date-like diagnosis column"

    for pattern, score in COLUMN_RULES.get(canonical, []):
        if re.search(pattern, key, flags=re.IGNORECASE):
            adjusted = score
            if key == canonical:
                adjusted += 50
            if canonical == "diagnosis" and key == "diagnosis":
                adjusted += 80
            if adjusted > best_score:
                best_score = adjusted
                best_rule = pattern
    return best_score, best_rule


def infer_metadata_column_mapping(df: pd.DataFrame, min_score: int = 70) -> tuple[dict[str, str], pd.DataFrame]:
    """Infer canonical metadata columns from flexible user-provided column names."""
    candidates: list[dict[str, Any]] = []
    used_original_cols: set[str] = set()
    mapping: dict[str, str] = {}

    ordered_canonicals = ["file_name", "extension", *[c for c in REQUIRED_METADATA_COLUMNS if c != "file_name"], *OPTIONAL_CANONICAL_COLUMNS]
    for canonical in ordered_canonicals:
        allow_reuse = canonical in OPTIONAL_CANONICAL_COLUMNS
        scored = []
        for col in df.columns:
            if (not allow_reuse) and col in used_original_cols:
                continue
            score, rule = _score_column(canonical, col)
            if score >= min_score:
                scored.append((score, col, rule))
        scored.sort(key=lambda x: (-x[0], str(x[1]).lower()))
        if scored:
            score, col, rule = scored[0]
            mapping[canonical] = col
            if not allow_reuse:
                used_original_cols.add(col)
            candidates.append({"canonical_column": canonical, "source_column": col, "score": score, "rule": rule, "selected": True})
        else:
            candidates.append({"canonical_column": canonical, "source_column": pd.NA, "score": 0, "rule": "", "selected": False})
    return mapping, pd.DataFrame(candidates)


def _apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for canonical, source in mapping.items():
        out[canonical] = df[source]
    return out


def _normalize_file_name_value(value: Any, extension: Any = None) -> str | None:
    if pd.isna(value):
        return None
    name = str(value).strip()
    if not name:
        return None
    ext = str(extension).strip() if extension is not None and not pd.isna(extension) else ""
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    if ext and Path(name).suffix == "":
        name = f"{name}{ext}"
    return Path(name).name


def _stem_key(file_name: str) -> str:
    return re.sub(r"\s+", "", Path(str(file_name)).stem).lower()


def _ext_priority(file_name: str) -> int:
    return EXTENSION_PRIORITY.get(Path(str(file_name)).suffix.lower(), 99)


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


def _prepare_demographics(path: Path, cfg: MetadataConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    raw = pd.read_csv(path)
    mapping, mapping_df = infer_metadata_column_mapping(raw)
    if "file_name" not in mapping:
        raise ValueError("Could not identify a file-name column in the metadata CSV. Please include a column such as Raw Media File name, file_name, filename, or audio file.")

    canon = _apply_column_mapping(raw, mapping)
    ext_series = canon["extension"] if "extension" in canon.columns else None
    canon["file_name"] = [
        _normalize_file_name_value(v, ext_series.iloc[i] if ext_series is not None else None)
        for i, v in enumerate(canon["file_name"])
    ]
    canon = canon[canon["file_name"].notna()].copy()
    canon["file_name_lc"] = canon["file_name"].astype(str).str.lower()
    canon["file_stem_lc"] = canon["file_name"].map(_stem_key)
    canon["metadata_extension_priority"] = canon["file_name"].map(_ext_priority)
    canon["metadata_row_index"] = canon.index.astype(int)

    for col in REQUIRED_METADATA_COLUMNS:
        if col not in canon.columns:
            canon[col] = pd.NA
            warnings.append(f"metadata CSV missing expected column '{col}'; column filled with NA")
    # severity fallback: if severity_score not mapped but ALSFRS total exists, use it.
    if canon["severity_score"].isna().all() and "alsfrs_total" in canon.columns:
        canon["severity_score"] = canon["alsfrs_total"]
        warnings.append("severity_score was populated from detected ALSFRS total column")

    if cfg.preserve_all_metadata_columns:
        for col in raw.columns:
            prefixed = f"meta_{_column_key(col)}"
            if prefixed not in canon.columns:
                canon[prefixed] = raw.loc[canon.index, col].values

    dup_exact = int(canon["file_name_lc"].duplicated().sum())
    if dup_exact:
        warnings.append(f"metadata CSV contains {dup_exact} duplicate exact file names; best extension-priority/first row retained for matching")

    return canon, mapping_df, warnings


def _normalize_task_value(value: Any, fallback_file_name: str | None = None) -> Any:
    if value is not None and not pd.isna(value):
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).upper().strip("_")
        tokens = [t for t in cleaned.split("_") if t]
        joined = "".join(tokens)
        for token in tokens:
            if token in TASK_ALIASES:
                return TASK_ALIASES[token]
        for alias, task in TASK_ALIASES.items():
            if alias in joined:
                return task
        if str(value).strip():
            return str(value).strip().lower().replace(" ", "_")
    if fallback_file_name:
        parsed, _ = _parse_task_from_name(fallback_file_name)
        return parsed if parsed else pd.NA
    return pd.NA


def _match_metadata_to_ingest(index_df: pd.DataFrame, demo_df: pd.DataFrame, cfg: MetadataConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demo_exact = demo_df.sort_values(["file_name_lc", "metadata_extension_priority", "metadata_row_index"]).drop_duplicates("file_name_lc", keep="first")
    exact_map = {str(r.file_name_lc): r for r in demo_exact.itertuples(index=False)}

    demo_stem = demo_df.sort_values(["file_stem_lc", "metadata_extension_priority", "metadata_row_index"]).drop_duplicates("file_stem_lc", keep="first")
    stem_map = {str(r.file_stem_lc): r for r in demo_stem.itertuples(index=False)}

    matched_rows: list[dict[str, Any]] = []
    linkage_rows: list[dict[str, Any]] = []
    used_meta_rows: set[int] = set()

    demo_cols = [c for c in demo_df.columns if c not in {"file_name_lc", "file_stem_lc", "metadata_extension_priority"}]

    for i, row in index_df.reset_index(drop=True).iterrows():
        file_name = str(row.get("file_name"))
        lc = file_name.lower()
        stem = _stem_key(file_name)
        match = None
        match_type = "none"
        confidence = 0.0
        if lc in exact_map:
            match = exact_map[lc]
            match_type = "exact_file_name"
            confidence = 1.0
        elif cfg.metadata_stem_match and stem in stem_map:
            match = stem_map[stem]
            match_type = "same_stem_different_extension"
            confidence = 0.92

        out = row.to_dict()
        if match is not None:
            meta = match._asdict()
            used_meta_rows.add(int(meta.get("metadata_row_index")))
            for col in demo_cols:
                val = meta.get(col)
                if col in REQUIRED_METADATA_COLUMNS or col in OPTIONAL_CANONICAL_COLUMNS or col.startswith("meta_"):
                    # Do not overwrite file_name from ingest; keep source metadata file name separately.
                    if col == "file_name":
                        out["metadata_file_name"] = val
                    else:
                        out[col] = val if not pd.isna(val) else out.get(col, pd.NA)
            out["metadata_source"] = "metadata_csv"
            out["metadata_status"] = "joined_from_csv"
            out["metadata_match_type"] = match_type
            out["metadata_match_confidence"] = confidence
            out["metadata_notes"] = f"linked to metadata row by {match_type}"
        else:
            out["metadata_status"] = "filename_parser_fallback"
            out["metadata_match_type"] = "none"
            out["metadata_match_confidence"] = 0.0
        matched_rows.append(out)
        linkage_rows.append({
            "file_name": file_name,
            "file_path": row.get("file_path"),
            "metadata_status": out.get("metadata_status"),
            "metadata_match_type": out.get("metadata_match_type"),
            "metadata_match_confidence": out.get("metadata_match_confidence"),
            "metadata_file_name": out.get("metadata_file_name", pd.NA),
            "subject_id": out.get("subject_id", pd.NA),
            "session_id": out.get("session_id", pd.NA),
            "iteration": out.get("iteration", pd.NA),
            "task": out.get("task", pd.NA),
        })

    unused = demo_df[~demo_df["metadata_row_index"].isin(used_meta_rows)].copy()
    return pd.DataFrame(matched_rows), pd.DataFrame(linkage_rows), unused


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
        base["metadata_match_type"] = "filename_only"
        base["metadata_match_confidence"] = 0.5 if cfg.allow_filename_parsing else 0.0
        rows.append(base)
    index_df = pd.DataFrame(rows)

    warnings: list[str] = []
    mapping_df = pd.DataFrame(columns=["canonical_column", "source_column", "score", "rule", "selected"])
    linkage_df = pd.DataFrame({"file_name": index_df.get("file_name", pd.Series(dtype=str))})
    unused_df = pd.DataFrame()
    if cfg.demographics_csv:
        demo_path = Path(cfg.demographics_csv).expanduser()
        demo_df, mapping_df, demo_warnings = _prepare_demographics(demo_path, cfg)
        warnings.extend(demo_warnings)
        index_df, linkage_df, unused_df = _match_metadata_to_ingest(index_df, demo_df, cfg)
        n_unmatched = int((index_df["metadata_match_type"] == "none").sum())
        if n_unmatched:
            warnings.append(f"{n_unmatched} ingested files were not linked to metadata CSV rows and used filename fallback")

    for col in [*REQUIRED_METADATA_COLUMNS, *OPTIONAL_CANONICAL_COLUMNS]:
        if col not in index_df.columns:
            index_df[col] = pd.NA

    # Dates and iteration normalization, without rejecting dirty data.
    if "recording_date" in index_df.columns:
        index_df["recording_date"] = pd.to_datetime(index_df["recording_date"], errors="coerce").dt.date.astype("string")
    if "assessment_date" in index_df.columns:
        index_df["assessment_date"] = pd.to_datetime(index_df["assessment_date"], errors="coerce").dt.date.astype("string")
    index_df["iteration"] = index_df["iteration"].astype("string")
    index_df["task"] = [
        _normalize_task_value(v, fallback_file_name=f)
        for v, f in zip(index_df["task"], index_df["file_name"])
    ]

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

    unmatched_df = linkage_df[linkage_df.get("metadata_match_type", "") == "none"].copy() if not linkage_df.empty else pd.DataFrame()
    unused_preview = unused_df.head(200).copy() if not unused_df.empty else unused_df

    index_path = folders["tables"] / "project_file_index.csv"
    completeness_path = folders["tables"] / "metadata_completeness.csv"
    mapping_path = folders["tables"] / "metadata_column_mapping.csv"
    linkage_path = folders["tables"] / "metadata_linkage_summary.csv"
    unmatched_path = folders["tables"] / "metadata_unmatched_ingested_files.csv"
    unused_path = folders["tables"] / "metadata_unused_rows_sample.csv"
    errors_path = folders["errors"] / "metadata_errors.csv"
    report_path = folders["reports"] / "metadata_report.html"

    index_df.to_csv(index_path, index=False)
    completeness_df.to_csv(completeness_path, index=False)
    mapping_df.to_csv(mapping_path, index=False)
    linkage_df.to_csv(linkage_path, index=False)
    unmatched_df.to_csv(unmatched_path, index=False)
    unused_preview.to_csv(unused_path, index=False)
    pd.DataFrame([]).to_csv(errors_path, index=False)
    _write_metadata_report(report_path, index_df, completeness_df, mapping_df, linkage_df, unused_df, warnings, cfg)

    status = "completed_with_warnings" if warnings else "completed"
    manifest = StageManifest(
        stage_name="acoustic_metadata",
        stage_version="0.2.0",
        status=status,
        output_artifacts=[
            ArtifactRef(path=str(index_path), role="project_file_index", media_type="text/csv"),
            ArtifactRef(path=str(completeness_path), role="metadata_completeness", media_type="text/csv"),
            ArtifactRef(path=str(mapping_path), role="metadata_column_mapping", media_type="text/csv"),
            ArtifactRef(path=str(linkage_path), role="metadata_linkage_summary", media_type="text/csv"),
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


def _write_metadata_report(
    path: Path,
    index_df: pd.DataFrame,
    completeness_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    linkage_df: pd.DataFrame,
    unused_df: pd.DataFrame,
    warnings: list[str],
    cfg: MetadataConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completeness_html = completeness_df.to_html(index=False, escape=True)
    selected_mapping_html = mapping_df[mapping_df.get("selected", False) == True].to_html(index=False, escape=True) if not mapping_df.empty else "<p>No metadata CSV supplied.</p>"
    status_counts = index_df["metadata_status"].value_counts(dropna=False).to_frame("count").reset_index().rename(columns={"index": "metadata_status"})
    status_html = status_counts.to_html(index=False, escape=True)
    match_counts = linkage_df["metadata_match_type"].value_counts(dropna=False).to_frame("count").reset_index().rename(columns={"index": "metadata_match_type"}).to_html(index=False, escape=True) if not linkage_df.empty else "<p>No linkage table.</p>"
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
<div class='card'><b>Files indexed:</b> {len(index_df)}<br><b>Demographics CSV:</b> {cfg.demographics_csv}<br><b>Unused metadata rows:</b> {len(unused_df)}</div>
<div class='card'><h2>Metadata source status</h2>{status_html}</div>
<div class='card'><h2>Linkage match types</h2>{match_counts}</div>
<div class='card'><h2>Detected column mapping</h2>{selected_mapping_html}</div>
<div class='card'><h2>Completeness</h2>{completeness_html}</div>
<div class='card'><h2>Warnings</h2><ul>{warning_html}</ul></div>
<div class='card'><p>Missing clinical labels are kept as NA. Filename-derived values are explicitly flagged.</p></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
