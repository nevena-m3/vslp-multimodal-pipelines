"""Measure MFA boundary accuracy against hand-corrected (gold) TextGrids.

The gold set is a folder of recordings, each with the automatic TextGrid
``<name>.TextGrid`` (as exported by the batch aligner) and the annotator's
corrected copy ``<name>_gold.TextGrid`` (saved from Praat as a text file; labels
are never renamed, only boundaries moved). A ``manifest.csv`` with at least
``file`` and optionally ``task`` and ``group`` columns adds the breakdown.

For every word, the onset and offset are compared; for vowel phones (ARPA
vowels) too. Output, in ``--out``:

* ``gold_boundaries.csv``: one row per boundary (auto, gold, error in ms);
* ``gold_summary.csv``: per tier, task and group: n, median and 90th
  percentile absolute error, and the share within 20 ms and 50 ms (the
  usual forced-alignment benchmarks);
* ``gold_accuracy.png``: error distributions per task and tier.

A recording whose gold labels do not match the automatic ones (a label was
changed or a word deleted) is listed with status LABEL_MISMATCH and left out,
never paired by guesswork.

    python -m vslp.acoustic.alignment.gold_eval --gold-dir G --out G\\results
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .mfa_provider import parse_textgrid

VOWEL = re.compile(r"^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[012]?$")


def compare_tier(auto: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame | None:
    """Boundary errors for two interval tables with the same labels in order."""
    if len(auto) != len(gold) or list(auto.label.str.casefold()) != list(gold.label.str.casefold()):
        return None
    rows = []
    for index, (a, g) in enumerate(zip(auto.itertuples(), gold.itertuples())):
        for edge in ("start_sec", "end_sec"):
            rows.append({"index": index, "label": a.label,
                         "edge": "onset" if edge == "start_sec" else "offset",
                         "auto_sec": float(getattr(a, edge)), "gold_sec": float(getattr(g, edge))})
    table = pd.DataFrame(rows)
    table["error_ms"] = (table.auto_sec - table.gold_sec) * 1000
    return table


def evaluate(gold_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold_dir = Path(gold_dir)
    manifest_path = gold_dir / "manifest.csv"
    manifest = (pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
                if manifest_path.is_file() else pd.DataFrame(columns=["file"]))
    info = {Path(row["file"]).stem: row for row in manifest.to_dict("records")}
    boundaries, status = [], []
    for gold_path in sorted(gold_dir.rglob("*_gold.TextGrid")):
        stem = gold_path.name[:-len("_gold.TextGrid")]
        auto_path = gold_path.with_name(f"{stem}.TextGrid")
        meta = info.get(stem, {})
        base = {"recording": stem, "task": meta.get("task", ""), "group": meta.get("group", "")}
        if not auto_path.is_file():
            status.append({**base, "status": "AUTO_TEXTGRID_MISSING"})
            continue
        auto_words, auto_phones = parse_textgrid(auto_path)
        gold_words, gold_phones = parse_textgrid(gold_path)
        words = compare_tier(auto_words, gold_words)
        if words is None:
            status.append({**base, "status": "LABEL_MISMATCH"})
            continue
        boundaries.append(words.assign(tier="word", **base))
        vowels = [table.loc[table.label.str.upper().str.match(VOWEL)].reset_index(drop=True)
                  if not table.empty else table for table in (auto_phones, gold_phones)]
        vowel_table = compare_tier(*vowels) if not vowels[0].empty else None
        if vowel_table is not None:
            boundaries.append(vowel_table.assign(tier="vowel", **base))
        status.append({**base, "status": "OK" if vowel_table is not None
                       else "OK_WORDS_ONLY"})
    table = pd.concat(boundaries, ignore_index=True) if boundaries else pd.DataFrame(
        columns=["recording", "task", "group", "tier", "edge", "error_ms"])
    return table, pd.DataFrame(status)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in [(("all", "all"), table)] + [
            ((task, "all"), part) for task, part in table.groupby("task")] + [
            (("all", group), part) for group, part in table.groupby("group")]:
        for tier, sub in part.groupby("tier"):
            error = sub.error_ms.abs()
            rows.append({"task": keys[0], "group": keys[1], "tier": tier,
                         "recordings": sub.recording.nunique(), "boundaries": len(sub),
                         "median_abs_ms": round(float(error.median()), 1),
                         "p90_abs_ms": round(float(error.quantile(.9)), 1),
                         "within_20ms_pct": round(100 * float((error <= 20).mean()), 1),
                         "within_50ms_pct": round(100 * float((error <= 50).mean()), 1),
                         "mean_signed_ms": round(float(sub.error_ms.mean()), 1)})
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = [tier for tier in ("word", "vowel") if tier in set(table.tier)]
    figure, axes = plt.subplots(1, max(1, len(tiers)), figsize=(6 * max(1, len(tiers)), 4.5),
                                squeeze=False)
    for axis, tier in zip(axes[0], tiers):
        part = table.loc[table.tier.eq(tier)]
        tasks = sorted(part.task.unique())
        data = [part.loc[part.task.eq(task), "error_ms"].clip(-200, 200) for task in tasks]
        axis.boxplot(data, vert=False, showfliers=True, flierprops={"markersize": 2})
        axis.set_yticks(range(1, len(tasks) + 1), [task or "(no task)" for task in tasks])
        for limit in (-20, 20):
            axis.axvline(limit, color="#2a9d8f", linestyle="--", linewidth=1)
        axis.axvline(0, color="#555", linewidth=.8)
        axis.set_xlabel("MFA minus hand boundary (ms, clipped at ±200)")
        axis.set_title(f"{tier} boundaries (dashed: ±20 ms)")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MFA accuracy against hand-corrected TextGrids")
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table, status = evaluate(args.gold_dir)
    table.to_csv(out / "gold_boundaries.csv", index=False)
    status.to_csv(out / "gold_status.csv", index=False)
    summary = summarise(table) if not table.empty else pd.DataFrame()
    summary.to_csv(out / "gold_summary.csv", index=False)
    if not table.empty:
        plot(table, out / "gold_accuracy.png")
    print(status.status.value_counts().to_string() if not status.empty else "no gold files")
    if not summary.empty:
        print(summary.loc[summary.task.eq("all") & summary.group.eq("all")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
