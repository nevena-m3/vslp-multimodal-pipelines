"""MFA accuracy against hand-corrected TextGrids."""
from __future__ import annotations

import pandas as pd

from vslp.acoustic.alignment.gold_eval import evaluate, summarise


def _grid(path, words, phones):
    rows = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "item [1]:", 'name = "words"']
    for index, (label, a, b) in enumerate(words, 1):
        rows += [f"intervals [{index}]:", f"xmin = {a}", f"xmax = {b}", f'text = "{label}"']
    rows += ["item [2]:", 'name = "phones"']
    for index, (label, a, b) in enumerate(phones, 1):
        rows += [f"intervals [{index}]:", f"xmin = {a}", f"xmax = {b}", f'text = "{label}"']
    path.write_text("\n".join(rows), encoding="utf-8")


def test_boundary_errors_and_label_mismatch(tmp_path):
    words = [("buy", 1.00, 1.20), ("bobby", 1.20, 1.60)]
    phones = [("B", 1.00, 1.05), ("AY1", 1.05, 1.20), ("B", 1.20, 1.25), ("AA1", 1.25, 1.40)]
    _grid(tmp_path / "r1.TextGrid", words, phones)
    _grid(tmp_path / "r1_gold.TextGrid", [("buy", 1.01, 1.23), ("bobby", 1.23, 1.60)],
          [("B", 1.01, 1.06), ("AY1", 1.06, 1.23), ("B", 1.23, 1.28), ("AA1", 1.28, 1.40)])
    _grid(tmp_path / "r2.TextGrid", words, phones)
    _grid(tmp_path / "r2_gold.TextGrid", [("bye", 1.0, 1.2), ("bobby", 1.2, 1.6)], phones)
    pd.DataFrame({"file": ["r1.wav", "r2.wav"], "task": ["s2", "s2"],
                  "group": ["ALS", "control"]}).to_csv(tmp_path / "manifest.csv", index=False)
    table, status = evaluate(tmp_path)
    assert dict(zip(status.recording, status.status)) == {"r1": "OK", "r2": "LABEL_MISMATCH"}
    words_only = table.loc[table.tier.eq("word")]
    assert sorted(words_only.error_ms.round(3)) == [-30.0, -30.0, -10.0, 0.0]
    summary = summarise(table)
    overall = summary.loc[summary.task.eq("all") & summary.group.eq("all") & summary.tier.eq("word")]
    assert overall.within_20ms_pct.iloc[0] == 50.0
    assert set(table.loc[table.tier.eq("vowel"), "label"]) == {"AY1", "AA1"}
