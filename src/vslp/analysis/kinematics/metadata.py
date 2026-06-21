"""Metadata loading and simple linking for the kinematics GUI."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

CANDIDATE_VIDEO_KEYS = ("video_id", "filename", "file", "video", "recording_id", "basename")
CANDIDATE_SUBJECT_KEYS = ("subject_id", "participant_id", "participant", "subj", "id")


def load_metadata(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def infer_link_columns(video_manifest: pd.DataFrame, metadata: pd.DataFrame) -> tuple[str | None, str | None, str]:
    meta_cols_lower = {c.lower(): c for c in metadata.columns}
    for key in CANDIDATE_VIDEO_KEYS:
        if key in meta_cols_lower and "video_id" in video_manifest.columns:
            return "video_id", meta_cols_lower[key], "video_id"
    for key in CANDIDATE_SUBJECT_KEYS:
        if key in meta_cols_lower:
            return None, meta_cols_lower[key], "subject_or_manual"
    return None, None, "manual_required"


def link_metadata(manifest_csv: Path, metadata_path: Path | None, output_dir: Path) -> dict[str, Path | int | str]:
    manifest = pd.read_csv(manifest_csv)
    out = Path(output_dir) / "kinematics" / "001_metadata" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    meta_out = out / "metadata_loaded.csv"
    linked_out = out / "metadata_link_preview.csv"
    if metadata_path is None:
        metadata = pd.DataFrame(columns=("metadata_status",))
        metadata.to_csv(meta_out, index=False)
        manifest.to_csv(linked_out, index=False)
        return {
            "metadata_loaded": meta_out,
            "link_preview": linked_out,
            "n_metadata_rows": 0,
            "link_mode": "no_metadata_provided",
        }

    metadata = load_metadata(metadata_path)
    video_col, meta_col, mode = infer_link_columns(manifest, metadata)
    metadata.to_csv(meta_out, index=False)
    if video_col and meta_col:
        linked = manifest.merge(metadata, left_on=video_col, right_on=meta_col, how="left", suffixes=("", "_meta"))
    else:
        linked = manifest.copy()
    linked.to_csv(linked_out, index=False)
    return {"metadata_loaded": meta_out, "link_preview": linked_out, "n_metadata_rows": len(metadata), "link_mode": mode}
