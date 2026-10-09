"""Human-readable evidence from the same final tokens exposed to features."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import soundfile as sf


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _page(audio: np.ndarray, rate: int, start: float, end: float,
          words: pd.DataFrame, phones: pd.DataFrame, intervals: pd.DataFrame,
          title: str) -> plt.Figure:
    a, b = max(0, round(start * rate)), min(len(audio), round(end * rate))
    signal = audio[a:b]
    times = np.arange(a, b) / rate
    fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, .8, 1.3]})
    fig.suptitle(title, fontsize=10)
    if signal.size:
        step = max(1, signal.size // 10000)
        axes[0].plot(times[::step], signal[::step], linewidth=.45)
        axes[1].specgram(signal, Fs=rate, NFFT=min(512, max(64, 2 ** int(
            np.floor(np.log2(max(64, signal.size)))))), noverlap=32, xextent=(a / rate, b / rate))
        axes[1].set_ylim(0, min(5000, rate / 2))
    for row in intervals.itertuples():
        left, right = float(row.start_sec), float(row.end_sec)
        if left < end and right > start:
            if str(row.segment_role) == "manual_exclusion":
                for axis in axes:
                    axis.axvspan(left, right, color="tomato", alpha=.22)
            elif str(row.segment_role) == "speech":
                axes[2].axvspan(left, right, color="seagreen", alpha=.5)
    axes[2].set_ylabel("Review")
    for axis, table, text_column in ((axes[2], words, "canonical_word_text"),
                                     (axes[3], phones, "phone_raw")):
        for token in table.itertuples():
            left, right = float(token.start_sec), float(token.end_sec)
            if left < end and right > start:
                axis.axvline(left, color="black", linewidth=.4)
                axis.text(max(left, start), .55, str(getattr(token, text_column)),
                          rotation=70, fontsize=6, va="center", clip_on=True)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
    axes[0].set_ylabel("Waveform")
    axes[1].set_ylabel("Hz")
    axes[3].set_ylabel("Phones")
    axes[3].set_xlim(start, end)
    axes[3].set_xlabel("Original recording time (s)")
    for axis in axes:
        axis.axvline(start, color="royalblue", linewidth=.7)
        axis.axvline(end, color="royalblue", linewidth=.7)
    fig.tight_layout()
    return fig


def create_final_evidence(root: Path, destination: Path, trials: pd.DataFrame,
                          words: pd.DataFrame, phones: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """One overview per frozen trial; Bamboo also receives paginated detail."""
    review = root / "acoustic" / "003_segmentation_review" / "final"
    decisions = pd.read_csv(review / "final_segmentation_decisions.csv", keep_default_na=False)
    intervals = pd.read_csv(review / "final_segmentation_intervals.csv", keep_default_na=False)
    by_record = decisions.set_index("recording_id")
    folder = destination / "evidence"
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for trial in trials.itertuples():
        identity, trial_id = str(trial.recording_id), str(trial.trial_id)
        audio, rate = sf.read(str(by_record.loc[identity].analysis_wav_path),
                              dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        word_rows = words.loc[words.recording_id.astype(str).eq(identity)
                              & words.trial_id.astype(str).eq(trial_id)]
        phone_rows = phones.loc[phones.recording_id.astype(str).eq(identity)
                                & phones.trial_id.astype(str).eq(trial_id)]
        review_rows = intervals.loc[intervals.recording_id.astype(str).eq(identity)]
        stem = f"{identity}_{trial_id}"
        overview = folder / f"{stem}_overview.png"
        stimulus = " ".join(word_rows.canonical_word_text.astype(str))
        title = (f"{identity} | {trial_id} | {trial.task_id}/{trial.task_item} | "
                 f"{trial.prompt_id} | {trial.alignment_review_status} | "
                 f"{trial.alignment_source_summary} | flags: {trial.qc_flags or 'none'} | "
                 f"{manifest.get('provider', '')} {manifest.get('provider_version', '')} | "
                 f"{manifest.get('acoustic_model', '')}/{manifest.get('dictionary', '')}\n"
                 f"Stimulus: {stimulus[:120]}{'…' if len(stimulus) > 120 else ''}")
        figure = _page(audio, rate, float(trial.start_sec), float(trial.end_sec),
                       word_rows, phone_rows, review_rows, title)
        figure.savefig(overview, dpi=140)
        plt.close(figure)
        detail = ""
        if str(trial.task_id) == "bamboo_passage":
            detail_path = folder / f"{stem}_detail.pdf"
            with PdfPages(detail_path) as pdf:
                for left in np.arange(float(trial.start_sec), float(trial.end_sec), 8.):
                    right = min(float(trial.end_sec), float(left) + 8.)
                    figure = _page(audio, rate, float(left), right, word_rows,
                                   phone_rows, review_rows, title)
                    pdf.savefig(figure)
                    plt.close(figure)
            detail = str(detail_path.relative_to(destination))
        index.append({"recording_id": identity, "trial_id": trial_id,
                      "alignment_run_id": str(trial.alignment_run_id),
                      "alignment_status": str(trial.alignment_review_status),
                      "overview_plot_path": str(overview.relative_to(destination)),
                      "detail_plot_path": detail,
                      "overview_sha256": _hash(overview),
                      "detail_sha256": _hash(destination / detail) if detail else ""})
    return pd.DataFrame(index)
