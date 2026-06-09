"""Report helpers for the kinematics GUI scaffold."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from .schemas import LANDMARK_PRESETS, NORMALIZATION_METHODS, AGGREGATION_PROFILES


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
    html = f"""
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
    <div class='card'><h2>Landmark presets</h2><table><tr><th>Preset</th><th>Indices</th><th>Count</th></tr>{''.join(f'<tr><td><b>{k}</b></td><td>{", ".join(map(str,v))}</td><td>{len(v)}</td></tr>' for k,v in LANDMARK_PRESETS.items())}</table></div>
    <div class='card'><h2>Normalization methods</h2><table><tr><th>Method</th><th>Interpretation</th></tr>{''.join(f'<tr><td><b>{k}</b></td><td>{v}</td></tr>' for k,v in NORMALIZATION_METHODS.items())}</table></div>
    <div class='card'><h2>Temporal aggregation profiles</h2><table><tr><th>Profile</th><th>Interpretation</th></tr>{''.join(f'<tr><td><b>{k}</b></td><td>{v}</td></tr>' for k,v in AGGREGATION_PROFILES.items())}</table></div>
    <div class='card warn'><h2>Scientific guardrails</h2><ul>
    <li>Video landmark trajectories are time-series measurements, not scalar biomarkers until cleaning, normalization, QC review, computation and aggregation are complete.</li>
    <li>MediaPipe tracking quality should be summarized using face-detected fraction, missing-frame burden, long gaps, landmark stability, pose/head motion, illumination, and interpolation burden; do not assume per-landmark confidence intervals exist.</li>
    <li>Normalization and temporal aggregation policies must be recorded because they directly define feature meaning.</li>
    <li>No clinical diagnosis, treatment recommendation, or automated patient-level decision is produced by this GUI.</li>
    </ul></div>
    </div></body></html>
    """
    path.write_text(html, encoding="utf-8")
    return path
