"""Report helpers for the kinematics GUI scaffold and pipeline outputs."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .schemas import LANDMARK_PRESETS, NORMALIZATION_METHODS, AGGREGATION_PROFILES


def _kin_root(output_root: Path) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    return output_root if output_root.name == "kinematics" else output_root / "kinematics"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _html_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "<p class='small'>No table available.</p>"
    safe = df.head(max_rows).copy()
    return safe.to_html(index=False, escape=True, border=0, classes="ref")


def _li(items):
    return "".join(items)


def write_scaffold_report(output_root: Path) -> Path:
    out = Path(output_root) / "kinematics" / "009_reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "kinematics_gui_workflow_outline_report.html"
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stages = [
        ("Setup / Ingest", "Discover/probe videos and create a source manifest."),
        ("Metadata", "Link subject, session, task, clinical, and acquisition context."),
        ("Face Landmarks", "Configure/run MediaPipe Face Landmarker extraction."),
        ("Landmark Selection", "Choose clinically meaningful landmark subsets."),
        ("Normalization", "Define coordinate scaling/stabilization policy."),
        ("Video QC", "Quantify face visibility, pose, illumination, stability, and task validity."),
        ("Feature Computation", "Compute cleaned, normalized frame/movement-level kinematic features."),
        ("Temporal Aggregation", "Collapse time series into transparent scalar summaries."),
        ("Data Inspector", "Review manifests, tables, configurations, and intermediate artifacts."),
        ("Reports & Outputs", "Package the run with provenance and SOP-aligned interpretation notes."),
    ]
    html_doc = f"""
    <html><head><meta charset='utf-8'><title>VSLP Kinematics GUI Workflow Outline</title>
    <style>
    body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f7fb;color:#172033}}
    .hero{{background:linear-gradient(135deg,#10233F,#1769AA);color:white;padding:34px 42px}}
    .wrap{{padding:28px 42px}}
    .card{{border:1px solid #dbe3ee;border-radius:16px;padding:18px;margin:16px 0;background:white;box-shadow:0 1px 2px rgba(16,35,63,.06)}}
    .grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}}
    .stage{{border-left:5px solid #1769AA;background:#fbfdff;border-radius:12px;padding:12px;border-top:1px solid #dbe3ee;border-right:1px solid #dbe3ee;border-bottom:1px solid #dbe3ee}}
    h1{{margin:0;font-size:30px}} h2{{color:#12345A;margin-top:0}} code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}
    .small{{color:#5f6f82;font-size:12px}} .warn{{background:#fff8e8;border-left:5px solid #d69b00}}
    table{{width:100%;border-collapse:collapse}} th,td{{border-bottom:1px solid #e2e8f0;padding:8px;text-align:left}} th{{background:#eaf1f8;color:#10233F}}
    </style></head>
    <body><div class='hero'><h1>VSLP Kinematics GUI Workflow Outline</h1>
    <p>Commercial scaffold for video-based facial/oral kinematic analysis using MediaPipe Face Landmarker.</p>
    <p class='small' style='color:#d9e9f8'>Generated {ts}. © 2026 Speech Production Lab, University of Toronto. Research software; not a clinical diagnostic device.</p></div>
    <div class='wrap'>
    <div class='card'><h2>Workflow</h2><div class='grid'>{''.join(f"<div class='stage'><b>{i+1}. {name}</b><br><span class='small'>{desc}</span></div>" for i,(name,desc) in enumerate(stages))}</div></div>
    <div class='card'><h2>Landmark presets</h2><table><tr><th>Preset</th><th>Indices</th><th>Count</th></tr>{''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(", ".join(map(str,v)))}</td><td>{len(v)}</td></tr>' for k,v in LANDMARK_PRESETS.items())}</table></div>
    <div class='card'><h2>Normalization methods</h2><table><tr><th>Method</th><th>Interpretation</th></tr>{''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(v)}</td></tr>' for k,v in NORMALIZATION_METHODS.items())}</table></div>
    <div class='card'><h2>Temporal aggregation profiles</h2><table><tr><th>Profile</th><th>Interpretation</th></tr>{''.join(f'<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(v)}</td></tr>' for k,v in AGGREGATION_PROFILES.items())}</table></div>
    <div class='card warn'><h2>Scientific guardrails</h2><ul>
    <li>Video landmark trajectories are time-series measurements, not scalar biomarkers until cleaning, normalization, QC review, computation and aggregation are complete.</li>
    <li>Video QC should include both automated landmark integrity checks and a future visual degradation framework: poor lighting, blur, freezing, camera instability, multiple people, occlusion, cropping, pose, and task compliance.</li>
    <li>Normalization and temporal aggregation policies must be recorded because they directly define feature meaning.</li>
    <li>No clinical diagnosis, treatment recommendation, or automated patient-level decision is produced by this GUI.</li>
    </ul></div>
    </div></body></html>
    """
    path.write_text(html_doc, encoding="utf-8")
    return path


def write_pipeline_summary_report(output_root: Path) -> dict[str, str]:
    """Write a current-output report using inspector tables if available."""
    kin = _kin_root(output_root)
    out = kin / "009_reports"
    out.mkdir(parents=True, exist_ok=True)
    report = out / "kinematics_pipeline_summary_report.html"
    manifest = out / "report_manifest.json"
    stage_df = _read_csv_if_exists(kin / "008_inspector" / "tables" / "stage_status.csv")
    artifact_df = _read_csv_if_exists(kin / "008_inspector" / "tables" / "artifact_inventory.csv")
    ingest_df = _read_csv_if_exists(kin / "000_ingest" / "tables" / "video_ingest_summary.csv")
    qc_df = _read_csv_if_exists(kin / "005_video_qc" / "tables" / "landmark_video_qc_summary.csv")
    feature_df = _read_csv_if_exists(kin / "006_features" / "tables" / "kinematic_features.csv")
    agg_df = _read_csv_if_exists(kin / "007_aggregation" / "tables" / "kinematic_aggregated_features.csv")

    def count_status(df: pd.DataFrame, col: str = "status") -> str:
        if df.empty or col not in df.columns:
            return "No table"
        return ", ".join(f"{k}: {v}" for k, v in df[col].value_counts(dropna=False).to_dict().items())

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    html_doc = f"""
    <html><head><meta charset='utf-8'><title>VSLP Kinematics Pipeline Summary</title>
    <style>
    body{{font-family:Segoe UI,Arial,sans-serif;background:#f6f8fb;color:#172033;margin:0}}
    .hero{{background:#10233F;color:#fff;padding:32px 42px}}
    .wrap{{padding:24px 42px}}
    .card{{background:#fff;border:1px solid #dbe3ee;border-radius:14px;padding:18px;margin:14px 0}}
    .grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
    .metric{{background:#f0f5fb;border:1px solid #dbe3ee;border-radius:12px;padding:12px}}
    .metric b{{display:block;color:#10233F;font-size:18px}}
    table.ref{{width:100%;border-collapse:collapse;font-size:12px}} .ref th,.ref td{{border-bottom:1px solid #e2e8f0;padding:7px;text-align:left;vertical-align:top}} .ref th{{background:#eaf1f8;color:#10233F}}
    .small{{color:#5f6f82;font-size:12px}} .warn{{border-left:5px solid #d69b00;background:#fff8e8}}
    h1{{margin:0}} h2{{margin-top:0;color:#12345A}}
    </style></head><body>
    <div class='hero'><h1>VSLP Kinematics Pipeline Summary</h1><p>Generated {html.escape(ts)}. Research output summary; not a diagnostic report.</p></div>
    <div class='wrap'>
      <div class='card'><h2>Run overview</h2><div class='grid'>
        <div class='metric'><span class='small'>Stages</span><b>{len(stage_df) if not stage_df.empty else 0}</b></div>
        <div class='metric'><span class='small'>Artifacts</span><b>{len(artifact_df) if not artifact_df.empty else 0}</b></div>
        <div class='metric'><span class='small'>QC status</span><b>{html.escape(count_status(qc_df))}</b></div>
        <div class='metric'><span class='small'>Aggregation status</span><b>{html.escape(count_status(agg_df))}</b></div>
      </div></div>
      <div class='card warn'><h2>Scientific guardrail</h2><p>This report packages pipeline outputs and provenance. It does not validate ALS, Parkinson's disease, or any clinical diagnosis. Interpret scalar features only with video QC, normalization QC, task context, and retained time-series review.</p></div>
      <div class='card'><h2>Stage status</h2>{_html_table(stage_df, 20)}</div>
      <div class='card'><h2>Ingest summary</h2>{_html_table(ingest_df, 20)}</div>
      <div class='card'><h2>Video / landmark QC summary</h2>{_html_table(qc_df, 25)}</div>
      <div class='card'><h2>Feature output preview</h2>{_html_table(feature_df, 15)}</div>
      <div class='card'><h2>Aggregated output preview</h2>{_html_table(agg_df, 15)}</div>
      <div class='card'><h2>Artifact inventory</h2>{_html_table(artifact_df[[c for c in ['stage','artifact_type','filename','relative_path','modified_utc','rows','columns'] if c in artifact_df.columns]] if not artifact_df.empty else artifact_df, 80)}</div>
    </div></body></html>
    """
    report.write_text(html_doc, encoding="utf-8")
    payload = {
        "schema": "vslp_kinematics_report_manifest_v1",
        "created_at_utc": ts,
        "pipeline_summary_report_html": str(report),
        "stage_status_csv": str(kin / "008_inspector" / "tables" / "stage_status.csv"),
        "artifact_inventory_csv": str(kin / "008_inspector" / "tables" / "artifact_inventory.csv"),
        "note": "Report is a provenance and output summary, not a clinical diagnostic interpretation.",
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"report_html": str(report), "manifest_json": str(manifest)}
