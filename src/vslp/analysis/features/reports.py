from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def _html_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No rows available.</p>"
    return df.head(max_rows).to_html(index=False, classes="tbl", border=0)


def write_report(output_root: Path, title: str, tables: Dict[str, Path], plots: Dict[str, Path], messages: Iterable[str] = ()) -> Path:
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "vslp_feature_analysis_report.html"
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:28px;background:#f8fafc;color:#111827;}
    h1,h2{color:#0b1f3a}.card{background:white;border:1px solid #d6dde8;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 6px 20px rgba(16,24,40,.05)}
    .muted{color:#667085}.tbl{border-collapse:collapse;width:100%;font-size:13px}.tbl th{background:#eef4ff;text-align:left}.tbl th,.tbl td{border-bottom:1px solid #e5e7eb;padding:6px 8px;}
    img{max-width:100%;border:1px solid #d6dde8;border-radius:10px;background:white;margin:8px 0}
    code{background:#eef4ff;padding:2px 5px;border-radius:4px}
    """
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>", f"<h1>{title}</h1>"]
    parts.append("<div class='card'><h2>Run notes</h2>")
    msgs = list(messages)
    if msgs:
        parts.append("<ul>" + "".join(f"<li>{m}</li>" for m in msgs) + "</ul>")
    else:
        parts.append("<p class='muted'>No warnings or notes were generated.</p>")
    parts.append("</div>")
    if plots:
        parts.append("<div class='card'><h2>Plots</h2>")
        for name, p in plots.items():
            rel = Path(p).relative_to(report_dir).as_posix() if Path(p).is_relative_to(report_dir) else Path(p).as_posix()
            # plots are usually ../plots/...
            try:
                rel = Path(p).relative_to(output_root).as_posix()
                rel = "../" + rel if not rel.startswith("reports/") else rel
            except Exception:
                rel = Path(p).as_posix()
            parts.append(f"<h3>{name}</h3><img src='{rel}' alt='{name}'>")
        parts.append("</div>")
    for name, p in tables.items():
        try:
            df = pd.read_csv(p)
        except Exception:
            df = pd.DataFrame()
        parts.append(f"<div class='card'><h2>{name}</h2><p><code>{p}</code></p>{_html_table(df)}</div>")
    parts.append("</body></html>")
    report.write_text("\n".join(parts), encoding="utf-8")
    return report
