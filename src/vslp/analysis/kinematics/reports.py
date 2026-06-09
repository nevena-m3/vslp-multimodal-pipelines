"""Report helpers for the kinematics GUI scaffold."""
from __future__ import annotations
from pathlib import Path
import json
from .schemas import LANDMARK_PRESETS, NORMALIZATION_METHODS, AGGREGATION_PROFILES


def write_scaffold_report(output_root: Path) -> Path:
    out = Path(output_root) / "kinematics" / "007_reports" 
    out.mkdir(parents=True, exist_ok=True)
    path = out / "kinematics_gui_scaffold_report.html"
    html = f"""
    <html><head><meta charset='utf-8'><title>VSLP Kinematics GUI Scaffold</title>
    <style>body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#1f2937}} .card{{border:1px solid #dbe3ee;border-radius:12px;padding:16px;margin:14px 0;background:#fbfdff}} code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}</style></head>
    <body><h1>VSLP Kinematics GUI Scaffold</h1>
    <p>This report summarizes the configured kinematic analysis workflow. Later patches will attach full landmark extraction, video QC, feature computation, scalar aggregation, and export artifacts.</p>
    <div class='card'><h2>Landmark presets</h2><ul>{''.join(f'<li><b>{k}</b>: {v}</li>' for k,v in LANDMARK_PRESETS.items())}</ul></div>
    <div class='card'><h2>Normalization methods</h2><ul>{''.join(f'<li><b>{k}</b>: {v}</li>' for k,v in NORMALIZATION_METHODS.items())}</ul></div>
    <div class='card'><h2>Aggregation profiles</h2><ul>{''.join(f'<li><b>{k}</b>: {v}</li>' for k,v in AGGREGATION_PROFILES.items())}</ul></div>
    <p><b>Guardrail:</b> frame-level landmark trajectories are not scalar biomarkers until cleaning, normalization, QC review, feature computation, and transparent temporal aggregation have been completed.</p>
    </body></html>
    """
    path.write_text(html, encoding="utf-8")
    return path
