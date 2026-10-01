"""Compact structural QC artifacts for private Alignment runs."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import soundfile as sf


INDEX_COLUMNS = (
    "alignment_run_id", "recording_id", "trial_id", "stimulus_id",
    "expected_repetitions", "observed_repetitions", "alignment_status",
    "review_mode", "review_status", "flags", "word_count_expected",
    "word_count_aligned", "phone_count", "alignment_source", "plot_path",
    "policy_version",
)


def _table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False) if path.is_file() else pd.DataFrame()


def _groups(frame: pd.DataFrame, *columns: str) -> dict[tuple[str, ...], pd.DataFrame]:
    if frame.empty:
        return {}
    return {tuple(str(value) for value in (key if isinstance(key, tuple) else (key,))): group
            for key, group in frame.groupby(list(columns), sort=False)}


def _waveform(path: Path, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    with sf.SoundFile(path) as wave:
        first, last = max(0, round(start * wave.samplerate)), min(
            wave.frames, round(end * wave.samplerate))
        wave.seek(first)
        frames = wave.read(max(0, last - first), dtype="float32", always_2d=True)
        mono = frames.mean(axis=1) if len(frames) else np.empty(0)
        stride = max(1, (len(mono) + 7999) // 8000)
        indices = np.arange(0, len(mono), stride)
        return (first + indices) / wave.samplerate, mono[::stride]


def _plot_trial(path: Path, trial: dict, audio: Path | None,
                exclusions: pd.DataFrame, words: pd.DataFrame, phones: pd.DataFrame,
                flags: list[str], status: str, source: str) -> None:
    fig = Figure(figsize=(12, 4), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    start, end = float(trial["start_sec"]), float(trial["end_sec"])
    if audio and audio.is_file():
        try:
            times, samples = _waveform(audio, start, end)
            ax.plot(times, samples, color="#407ca4", linewidth=.55)
        except (OSError, ValueError, RuntimeError):
            ax.text(.5, .6, "Audio display unavailable", transform=ax.transAxes,
                    ha="center")
    else:
        ax.text(.5, .6, "Audio unavailable", transform=ax.transAxes, ha="center")
    ax.axvline(start, color="#2d5b9a", linewidth=1)
    ax.axvline(end, color="#2d5b9a", linewidth=1)
    for row in exclusions.itertuples():
        left, right = float(row.start_sec), float(row.end_sec)
        if left < end and right > start:
            ax.axvspan(max(start, left), min(end, right), color="#b64b4b", alpha=.3)
    for tier, frame, y, color, label_col in (
            ("word", words, -.34, "#b88321", "word"),
            ("phone", phones, -.68, "#31815c", "phone")):
        for row in frame.itertuples():
            left, right = float(row.start_sec), float(row.end_sec)
            if right <= start or left >= end:
                continue
            ax.hlines(y, left, right, color=color, linewidth=7, alpha=.45)
            ax.text((left + right) / 2, y + .035, str(getattr(row, label_col)),
                    ha="center", va="bottom", fontsize=6, rotation=0)
    ax.set_xlim(start, end)
    ax.set_ylim(-.82, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("Original recording time (s)")
    ax.set_title(f"{trial['recording_id']} · {trial['trial_id']} · "
                 f"{trial.get('prompt_id', '')} · {status} · {source}", fontsize=10)
    fig.text(.02, .02, "Flags: " + (", ".join(flags) if flags else "none"), fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.clear()


def _plot_summary(path: Path, frame: pd.DataFrame) -> None:
    flags = [flag for raw in frame["flags"] for flag in json.loads(raw)] if not frame.empty else []
    from collections import Counter
    counts = Counter(flags)
    headline = {
        "recordings": int(frame.recording_id.nunique()) if not frame.empty else 0,
        "trials": len(frame),
        "auto-accepted": int(frame.review_status.eq("AUTO_ACCEPTED_STRUCTURAL").sum()),
        "human accepted": int(frame.review_status.eq("ACCEPTED").sum()),
        "needs review": int(frame.review_status.eq("NEEDS_REVIEW").sum()),
        "failed": int(frame.alignment_status.ne("ALIGNED").sum()),
    }
    labels = list(headline) + list(counts)
    values = list(headline.values()) + list(counts.values())
    fig = Figure(figsize=(max(8, .65 * len(labels)), 4), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.bar(range(len(labels)), values, color=["#4c789a"] * len(headline) +
           ["#bb7748"] * len(counts))
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Alignment structural QC — engineering review counts")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.clear()


def create_alignment_qc(root: str | Path, run_dir: str | Path, manifest: dict,
                        *, final_dir: str | Path | None = None,
                        plot_all: bool = False,
                        selected_trial: tuple[str, str] | None = None) -> dict[str, str]:
    """Create a summary, index, and plots for flagged cases; no scientific score."""
    root, run_dir = Path(root), Path(run_dir)
    destination = Path(final_dir) if final_dir else run_dir
    plan_path = run_dir / "configs" / "alignment_trials.json"
    if not plan_path.is_file():
        return {}
    trials = json.loads(plan_path.read_text(encoding="utf-8"))["trials"]
    prefix = "final_" if final_dir else ""
    tables = destination / "tables" if not final_dir else destination
    reviews = _table(tables / f"{prefix}alignment_trial_review.csv")
    diagnostics = _table((run_dir / "tables" / "alignment_trial_diagnostics.csv"))
    recording_diagnostics = _table(run_dir / "tables" / "alignment_diagnostics.csv")
    links = _table(tables / f"{prefix}alignment_trial_tokens.csv")
    words = _table(tables / f"{prefix}alignment_words.csv")
    phones = _table(tables / f"{prefix}alignment_phones.csv")
    decisions = _table(root / "acoustic" / "003_segmentation_review" / "final" /
                       "final_segmentation_decisions.csv")
    intervals = _table(root / "acoustic" / "003_segmentation_review" / "final" /
                       "final_segmentation_intervals.csv")
    by_trial_diagnostics = _groups(diagnostics, "recording_id", "trial_id")
    by_record_diagnostics = _groups(recording_diagnostics, "recording_id")
    by_review = _groups(reviews, "recording_id", "trial_id")
    by_links = _groups(links, "recording_id", "trial_id")
    by_words = _groups(words, "recording_id")
    by_phones = _groups(phones, "recording_id")
    by_decisions = _groups(decisions, "recording_id")
    by_intervals = _groups(intervals, "recording_id")
    index_path = tables / f"{prefix}alignment_qc_index.csv"
    existing_index = _table(index_path) if selected_trial else pd.DataFrame()
    previous_plots = ({(str(row.recording_id), str(row.trial_id)): str(row.plot_path)
                       for row in existing_index.itertuples()}
                      if not existing_index.empty else {})
    index_rows = []
    for trial in trials:
        identity, trial_id = str(trial["recording_id"]), str(trial["trial_id"])
        chosen = by_trial_diagnostics.get((identity, trial_id))
        record = by_record_diagnostics.get((identity,))
        review = by_review.get((identity, trial_id))
        row = chosen.iloc[0] if chosen is not None else None
        record_row = record.iloc[0] if record is not None else None
        review_row = review.iloc[0] if review is not None else None
        flags = json.loads(review_row.flags_triggered) if review_row is not None and str(
            review_row.get("flags_triggered", "")).startswith("[") else []
        if row is None:
            flags.append(str(record_row.reason or record_row.status) if record_row is not None
                         else "ALIGNMENT_FAILED")
        status = str(row.status) if row is not None else "ALIGNMENT_FAILED"
        review_status = str(review_row.review_status) if review_row is not None else "NEEDS_REVIEW"
        review_mode = str(review_row.get("review_mode", "")) if review_row is not None else ""
        tokens = by_links.get((identity, trial_id))
        record_words = by_words.get((identity,))
        record_phones = by_phones.get((identity,))
        selected_words = (record_words.loc[record_words.word_index.isin(tokens.loc[
            tokens.token_type.eq("word"), "token_index"])]
            if record_words is not None and tokens is not None else words.iloc[0:0])
        selected_phones = (record_phones.loc[record_phones.phone_index.isin(tokens.loc[
            tokens.token_type.eq("phone"), "token_index"])]
            if record_phones is not None and tokens is not None else phones.iloc[0:0])
        source = (str(selected_words.alignment_source.iloc[0]) if not selected_words.empty
                  else manifest.get("source", ""))
        plot = previous_plots.get((identity, trial_id), "")
        should_plot = (selected_trial == (identity, trial_id) if selected_trial else
                       bool(flags) or review_status not in {
                           "AUTO_ACCEPTED_STRUCTURAL", "ACCEPTED"} or plot_all)
        if should_plot:
            safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{identity}_{trial_id}")
            plot_path = destination / "diagnostics" / "flagged_trials" / f"{safe_name}.png"
            decision = by_decisions.get((identity,))
            audio = Path(str(decision.iloc[0].analysis_wav_path)) if decision is not None else None
            record_intervals = by_intervals.get((identity,))
            excluded = (record_intervals.loc[record_intervals.segment_role.eq(
                "manual_exclusion")] if record_intervals is not None else intervals.iloc[0:0])
            _plot_trial(plot_path, trial, audio, excluded, selected_words, selected_phones,
                        flags, review_status, source)
            plot = str(plot_path.relative_to(destination))
        index_rows.append({
            "alignment_run_id": manifest["alignment_run_id"], "recording_id": identity,
            "trial_id": trial_id, "stimulus_id": trial.get("prompt_id", ""),
            "expected_repetitions": manifest.get("expected_trials_by_recording", {}).get(identity, ""),
            "observed_repetitions": manifest.get("observed_trials_by_recording", {}).get(identity, ""),
            "alignment_status": status, "review_mode": review_mode,
            "review_status": review_status, "flags": json.dumps(sorted(set(flags))),
            "word_count_expected": int(row.expected_word_count) if row is not None else 0,
            "word_count_aligned": int(row.n_words) if row is not None else 0,
            "phone_count": int(row.n_phones) if row is not None else 0,
            "alignment_source": source, "plot_path": plot,
            "policy_version": manifest.get("alignment_review_policy_version", ""),
        })
    index = pd.DataFrame(index_rows, columns=INDEX_COLUMNS)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False)
    summary_path = destination / "diagnostics" / "alignment_qc_summary.png"
    _plot_summary(summary_path, index)
    return {"qc_index_path": str(index_path), "qc_summary_plot_path": str(summary_path)}


def generate_selected_trial_plot(root: str | Path, run_id: str,
                                 recording_id: str, trial_id: str) -> Path:
    """Generate one optional review plot without plotting all accepted trials."""
    root = Path(root)
    run = root / "acoustic" / "004_alignment" / "runs" / run_id
    manifest = json.loads((run / "logs" / "stage_manifest.json").read_text(
        encoding="utf-8"))
    artifacts = create_alignment_qc(root, run, manifest,
                                    selected_trial=(recording_id, trial_id))
    manifest["alignment_qc_artifacts"] = {
        key: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for key, path in artifacts.items()}
    (run / "logs" / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    index = pd.read_csv(run / "tables" / "alignment_qc_index.csv",
                        keep_default_na=False)
    chosen = index.loc[index.recording_id.astype(str).eq(recording_id)
                       & index.trial_id.astype(str).eq(trial_id)]
    if len(chosen) != 1 or not chosen.iloc[0].plot_path:
        raise ValueError("alignment_trial_not_found")
    return run / chosen.iloc[0].plot_path
