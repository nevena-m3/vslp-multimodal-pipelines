"""Excel review sheet for recordings that alignment QC flagged (NEEDS_REVIEW).

Reads one or more per-recording tables (``batch_recordings.csv`` from the
batch aligner, ``sit_recordings.csv`` from the SIT aligner) and writes
``review_sheet.xlsx`` with:

* **Summary**: flagged recordings per reason and task, so common causes can
  be judged together before anyone listens to single files;
* **Review**: one row per flagged recording, sorted by reason, with a link to
  its picture (waveform, spectrogram, words, phones), an empty ``decision``
  column (ACCEPT / REJECT / FIX, picked from a list) and ``notes``.

Nothing is decided by this script; it only lays the work out.

    python -m vslp.acoustic.alignment.review_sheet --tables W\\batch_recordings.csv \
        W\\sit_alignment\\sit_recordings.csv --plots W\\plots W\\sit_alignment\\plots \
        --out W\\review_sheet.xlsx [--groups cohort.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

DECISIONS = ("ACCEPT", "REJECT", "FIX")


def flag_reason(flags: str) -> str:
    """The first QC flag without its detail (``words_not_in_dictionary:X`` -> name)."""
    first = next((part for part in str(flags).split(";") if part.strip()), "")
    return first.split(":", 1)[0].strip() or "(no flag)"


def _task(row: pd.Series) -> str:
    job = str(row.get("job", "") or "")
    if job:
        return job.split("__")[0]
    return "1059_SIT" if "_SIT" in str(row.get("file_name", "")).upper() else ""


def collect(tables: list[str | Path], plot_dirs: list[str | Path],
            groups: pd.DataFrame | None = None) -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in tables]
    data = (pd.concat(frames, ignore_index=True).fillna("") if frames else pd.DataFrame())
    if data.empty:
        return pd.DataFrame(columns=["task", "reason", "file_name", "group", "qc_flags",
                                     "plot", "decision", "notes"])
    flagged = data.loc[data.state.ne("AUTO_PASS")].copy()
    pictures: dict[str, Path] = {}
    for folder in plot_dirs:
        for path in Path(folder).rglob("*.png"):
            pictures.setdefault(path.stem, path.resolve())
    flagged["task"] = flagged.apply(_task, axis=1)
    flagged["reason"] = flagged.qc_flags.map(flag_reason)
    flagged["plot"] = [str(pictures.get(Path(name).stem, "")) for name in flagged.file_name]
    subject = flagged.get("speaker_id", pd.Series([""] * len(flagged), index=flagged.index))
    flagged["group"] = ""
    if groups is not None and {"subject_id", "group"} <= set(groups.columns):
        lookup = dict(zip(groups.subject_id.astype(str), groups.group.astype(str)))
        flagged["group"] = [lookup.get(str(value), "") for value in subject]
    flagged["decision"] = ""
    flagged["notes"] = ""
    return (flagged.sort_values(["reason", "task", "file_name"])
            [["task", "reason", "file_name", "group", "qc_flags", "plot", "decision",
              "notes"]].reset_index(drop=True))


def summary(review: pd.DataFrame) -> pd.DataFrame:
    if review.empty:
        return pd.DataFrame(columns=["reason", "total"])
    table = pd.crosstab(review.reason, review.task)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False).reset_index()


def write(review: pd.DataFrame, destination: str | Path) -> Path:
    """Write the .xlsx; without openpyxl, two CSVs next to it instead."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from openpyxl.styles import Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        summary(review).to_csv(destination.with_name(destination.stem + "_summary.csv"),
                               index=False)
        target = destination.with_suffix(".csv")
        review.to_csv(target, index=False)
        return target
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        summary(review).to_excel(writer, sheet_name="Summary", index=False)
        review.to_excel(writer, sheet_name="Review", index=False)
        sheet = writer.sheets["Review"]
        columns = {name: index + 1 for index, name in enumerate(review.columns)}
        plot_col, decision_col = columns["plot"], columns["decision"]
        for row in range(2, len(review) + 2):
            cell = sheet.cell(row=row, column=plot_col)
            if cell.value:
                cell.hyperlink = Path(str(cell.value)).as_uri()
                cell.value = "open picture"
                cell.font = Font(color="0563C1", underline="single")
        if len(review):
            letter = sheet.cell(row=1, column=decision_col).column_letter
            rule = DataValidation(type="list", formula1='"' + ",".join(DECISIONS) + '"',
                                  allow_blank=True)
            sheet.add_data_validation(rule)
            rule.add(f"{letter}2:{letter}{len(review) + 1}")
            fill = PatternFill("solid", fgColor="FFF2CC")
            for row in range(2, len(review) + 2):
                sheet.cell(row=row, column=decision_col).fill = fill
        widths = {"task": 22, "reason": 40, "file_name": 46, "group": 10, "qc_flags": 60,
                  "plot": 14, "decision": 12, "notes": 40}
        for name, index in columns.items():
            sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = (
                widths.get(name, 14))
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--plots", nargs="*", default=[])
    parser.add_argument("--groups", help="CSV with subject_id and group (e.g. ALS/PLS/control)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    groups = (pd.read_csv(args.groups, dtype=str, keep_default_na=False)
              if args.groups else None)
    review = collect(args.tables, args.plots, groups)
    target = write(review, args.out)
    print(f"{len(review)} flagged recordings -> {target}")
    print(summary(review).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
