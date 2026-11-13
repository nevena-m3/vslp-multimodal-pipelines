"""MFA alignment of SIT recordings, each with its own transcript.

In the SIT (1059_SIT) every participant reads ten different sentences, so no
fixed prompt exists. The text for MFA is that recording's own speech-to-text
transcript (the SIT screening JSON). Examiner item numbers ("5A", "number 6")
are not part of the reading; each is replaced by ``<unk>``, MFA's garbage word
(one ``spn`` phone), so the examiner's voice is absorbed instead of stretching
the neighbouring words.

This path does not produce GUI alignment packages: the package contract binds
one registered prompt per task, and SIT has none. It writes, in ``--out``:

* ``textgrids/<stem>.TextGrid``: MFA output in original recording time
  (whole recording, no crop);
* ``sit_words.csv`` and ``sit_phones.csv``: every word / phone with times;
* ``sit_recordings.csv``: one row per recording with state AUTO_PASS or
  NEEDS_REVIEW and the reasons;
* ``plots/<stem>.png`` for every NEEDS_REVIEW recording (``--plots all`` for
  all).

A recording is held out of MFA (NEEDS_REVIEW, never guessed) when its
transcript is empty, still holds digits after examiner numbers are removed, or
has words missing from the pronunciation dictionary. One bad transcript never
stops the batch.

    python -m vslp.acoustic.alignment.sit_align --screen-csv S\\sit_screen.csv \
        --json-dir S\\json --audio-dir S\\audio --out W\\sit_alignment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import pandas as pd

from .contract import normalize_transcript
from .mfa_provider import MfaProfile, MfaProvider, dictionary_oovs, parse_textgrid

EXAMINER = "<unk>"
# "5A", "5 A.", "10b", "number 6", "#7": item numbers the examiner reads out.
_NUMBER_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                 "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty")
# Examiner item labels: "5A", "5 A.", "10b", "number 6", "#7", and Whisper's
# spelled forms "five A.", "five-a", "elevenee". A spaced letter counts only
# before punctuation, so the article in "5 a boy ran" is kept.
EXAMINER_PATTERN = re.compile(
    r"(?:\bnumber\s+|#\s*)\d{1,2}[a-d]?\b\.?"
    r"|\b\d{1,2}(?:[a-d]\b|\s[a-d](?=[.,:;)]))\.?"
    rf"|\b(?:{_NUMBER_WORDS})(?:-?(?:a|ay|ee+)\b|\s(?:a|ay)(?=[.,:;)]))\.?",
    flags=re.IGNORECASE)
WORD_SPAN_REVIEW_SEC = 2.0
LOW_ASR_CONFIDENCE = -0.6  # Whisper average log-probability per token
DEFAULT_PROFILE = (Path(__file__).parent / "profiles" /
                   "mfa_vslp_en_us_arpa_334_wide_beam_v1.json")


def transcript_text(data) -> str:
    """Whole transcript from a screening JSON, whatever Whisper layout it uses."""
    if isinstance(data, list):
        return " ".join(transcript_text(item) for item in data).strip()
    if not isinstance(data, dict):
        return str(data or "").strip()
    for key in ("whole_text", "text"):
        if isinstance(data.get(key), str) and data[key].strip():
            return data[key].strip()
    for key in ("chunks", "segments"):
        if isinstance(data.get(key), list) and data[key]:
            return " ".join(str(item.get("text", "")).strip()
                            for item in data[key] if isinstance(item, dict)).strip()
    if isinstance(data.get("words"), list):
        return " ".join(str(item.get("word", item.get("text", ""))).strip()
                        for item in data["words"] if isinstance(item, dict)).strip()
    return ""


def mfa_transcript(text: str, examiner: str = "unk") -> tuple[str, list[str]]:
    """Text for the .lab file and the reasons it cannot be used ([] if fine).

    The examiner usually opens with a spoken preamble (date, participant ID,
    visit), which is not part of the reading. Everything before the first item
    label ("5A") is replaced by one ``<unk>`` (or dropped with ``drop``), so its
    numbers neither block the recording nor stretch the first sentence.
    """
    marker = "\x00"
    first = EXAMINER_PATTERN.search(text)
    preamble = bool(first and text[:first.start()].strip())
    if first:
        text = text[first.start():]
    marked = EXAMINER_PATTERN.sub(f" {marker} ", text)
    pieces = []
    for part in marked.split(marker):
        try:
            pieces.append(normalize_transcript(part))
        except ValueError:
            return "", ["transcript_contains_number_requires_review"]
    joiner = f" {EXAMINER} " if examiner == "unk" else " "
    tokens = joiner.join(pieces).split()
    # A label at the very start or end has nothing to absorb (silence covers
    # it); a dropped preamble does: the examiner was talking there.
    while tokens and tokens[0] == EXAMINER:
        tokens.pop(0)
    while tokens and tokens[-1] == EXAMINER:
        tokens.pop()
    if not tokens:
        return "", ["empty_transcript"]
    if preamble and examiner == "unk":
        tokens.insert(0, EXAMINER)
    return " ".join(tokens), []


def _speaker(file_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", Path(file_name).stem.split("_")[0]) or "unknown"


def prepare(screen: pd.DataFrame, json_dir: Path, audio_dir: Path, corpus: Path,
            dictionary: Path | None, examiner: str = "unk") -> pd.DataFrame:
    """Write the MFA corpus; return one row per recording with its pre-MFA state."""
    rows = []
    for item in screen.itertuples():
        name = str(item.file)
        stem = Path(name).stem
        row = {"file_name": name, "speaker_id": _speaker(name), "transcript": "",
               "flags": [], "avg_logprob": getattr(item, "avg_logprob", "")}
        source = audio_dir / name
        json_path = json_dir / f"{stem}.json"
        if not source.is_file():
            row["flags"].append("audio_missing")
        elif not json_path.is_file():
            row["flags"].append("transcript_missing")
        else:
            text = transcript_text(json.loads(json_path.read_text(encoding="utf-8")))
            lab, problems = mfa_transcript(text, examiner)
            row["flags"].extend(problems)
            if lab and dictionary is not None:
                oovs = dictionary_oovs(dictionary, [" ".join(
                    token for token in lab.split() if token != EXAMINER)])
                if oovs:
                    row["flags"].append("words_not_in_dictionary:" + "|".join(oovs))
            if not row["flags"]:
                speaker = corpus / f"speaker_{row['speaker_id']}"
                speaker.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, speaker / f"{stem}.wav")
                (speaker / f"{stem}.lab").write_text(lab, encoding="utf-8")
                row["transcript"] = lab
        rows.append(row)
    return pd.DataFrame(rows)


def check_alignment(row: dict, grid: Path | None, duration_sec: float) -> tuple[
        list[str], pd.DataFrame, pd.DataFrame]:
    """Structural QC of one SIT TextGrid against its own transcript."""
    empty = pd.DataFrame(columns=["label", "start_sec", "end_sec"])
    if grid is None or not grid.is_file():
        return ["mfa_output_missing"], empty, empty
    words, phones = parse_textgrid(grid)
    flags = []
    expected = [token for token in row["transcript"].split() if token != EXAMINER]
    lexical = (words.loc[~words.label.astype(str).str.casefold().eq(EXAMINER)]
               if not words.empty else words)
    actual = [normalize_transcript(str(label)) for label in lexical.get("label", [])]
    if actual != expected:
        flags.append("transcript_token_mismatch")
    if lexical.empty or phones.empty:
        flags.append("empty_alignment_tier")
    else:
        spans = lexical.end_sec.astype(float) - lexical.start_sec.astype(float)
        if (spans > WORD_SPAN_REVIEW_SEC).any():
            flags.append("long_aligned_word_span_requires_review")
        if float(lexical.end_sec.astype(float).max()) > duration_sec + 1e-6:
            flags.append("invalid_alignment_bounds")
    confidence = pd.to_numeric(pd.Series([row.get("avg_logprob")]), errors="coerce").iloc[0]
    if pd.notna(confidence) and confidence < LOW_ASR_CONFIDENCE:
        # Whisper unsure: the transcript itself may hold a misheard word.
        flags.append("low_speech_to_text_confidence")
    return flags, lexical, phones


def run(*, screen_csv: str | Path, json_dir: str | Path, audio_dir: str | Path,
        out: str | Path, profile_path: str | Path = DEFAULT_PROFILE, provider=None,
        usable_only: bool = True, examiner: str = "unk", plots: str = "flagged",
        log=print) -> pd.DataFrame:
    import soundfile as sf

    out = Path(out).resolve()
    corpus, mfa_out, logs = out / "corpus", out / "mfa_output", out / "logs"
    for folder in (corpus, mfa_out):
        if folder.exists():
            shutil.rmtree(folder)  # a rerun starts clean; sources are never touched
    corpus.mkdir(parents=True)
    screen = pd.read_csv(screen_csv, dtype=str, keep_default_na=False)
    if usable_only and "looks_like_reading" in screen:
        kept = screen.looks_like_reading.str.casefold().isin({"true", "1", "yes"})
        log(f"{int(kept.sum())} of {len(screen)} SIT recordings look like sentence reading")
        screen = screen.loc[kept]
    profile = MfaProfile.load(profile_path)
    provider = provider or MfaProvider()
    if hasattr(provider, "inspect_environment"):
        provider.inspect_environment(profile)  # resolves the dictionary path
    dictionary = getattr(provider, "resolved_dictionary", None)
    rows = prepare(screen, Path(json_dir), Path(audio_dir), corpus,
                   Path(dictionary) if dictionary else None, examiner)
    ready = rows.loc[rows["flags"].map(len).eq(0)]
    log(f"{len(ready)} recordings go to MFA; {len(rows) - len(ready)} held for review")
    grids: dict[str, Path] = {}
    if len(ready):
        preflight = provider.preflight(corpus, profile, [
            " ".join(t for t in text.split() if t != EXAMINER) for text in ready.transcript],
            logs)
        if preflight.get("status") != "AVAILABLE":
            log(f"MFA validation failed: {preflight.get('status')}")
            failed = f"alignment_not_completed:{preflight.get('status')}"
            rows["flags"] = [flags or [failed] for flags in rows["flags"]]
        else:
            provider.configure_from_preflight(profile, preflight)
            log("Aligning (one corpus-level MFA run)...")
            grids = provider.align_corpus(corpus, mfa_out, profile, logs)
    (out / "textgrids").mkdir(parents=True, exist_ok=True)
    word_tables, phone_tables, records = [], [], []
    for row in rows.to_dict("records"):
        stem = Path(row["file_name"]).stem
        flags = list(row["flags"])
        words = phones = pd.DataFrame()
        if not flags:
            duration = float(sf.info(Path(audio_dir) / row["file_name"]).duration)
            qc, words, phones = check_alignment(row, grids.get(stem), duration)
            flags.extend(qc)
            if grids.get(stem):
                shutil.copy2(grids[stem], out / "textgrids" / f"{stem}.TextGrid")
        for table, store in ((words, word_tables), (phones, phone_tables)):
            if not table.empty:
                store.append(table.assign(file_name=row["file_name"],
                                          index=range(1, len(table) + 1)))
        state = "NEEDS_REVIEW" if flags else "AUTO_PASS"
        records.append({"file_name": row["file_name"], "speaker_id": row["speaker_id"],
                        "state": state, "qc_flags": ";".join(flags),
                        "aligned_words": len(words), "transcript": row["transcript"],
                        "textgrid_path": str(out / "textgrids" / f"{stem}.TextGrid")
                        if grids.get(stem) else ""})
        if plots != "none" and (plots == "all" or state != "AUTO_PASS"):
            try:
                plot_sit(Path(audio_dir) / row["file_name"], words, phones,
                         out / "plots" / f"{stem}.png",
                         f"{row['file_name']} · {state} · {';'.join(flags)}")
            except Exception as exc:  # noqa: BLE001 - plots never block results
                log(f"plot failed for {row['file_name']}: {exc}")
    columns = ["file_name", "index", "label", "start_sec", "end_sec"]
    for name, tables in (("sit_words.csv", word_tables), ("sit_phones.csv", phone_tables)):
        (pd.concat(tables, ignore_index=True)[columns] if tables
         else pd.DataFrame(columns=columns)).to_csv(out / name, index=False)
    table = pd.DataFrame(records)
    table.to_csv(out / "sit_recordings.csv", index=False)
    log(table.state.value_counts().to_string() if len(table) else "no SIT recordings")
    return table


def plot_sit(source: Path, words: pd.DataFrame, phones: pd.DataFrame, destination: Path,
             title: str) -> Path:
    """Waveform, spectrogram, word and phone tiers (original recording time)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    duration = len(mono) / rate
    figure, axes = plt.subplots(4, 1, figsize=(16, 7), sharex=True,
                                gridspec_kw={"height_ratios": [1.2, 2, .6, .6]})
    step = max(1, len(mono) // 200000)
    axes[0].plot((np.arange(len(mono)) / rate)[::step], mono[::step], color="#555", lw=.4)
    axes[0].set_yticks([])
    with np.errstate(divide="ignore"):
        _s, _f, _b, image = axes[1].specgram(mono, NFFT=512, Fs=rate, noverlap=384,
                                             cmap="magma")
    finite = image.get_array()[np.isfinite(image.get_array())]
    if finite.size:
        image.set_clim(float(finite.max()) - 70, float(finite.max()))
    axes[1].set_ylim(0, 8000)
    axes[1].set_ylabel("Hz")
    for axis, table, color, size in ((axes[2], words, "#2a6fdb", 7),
                                     (axes[3], phones, "#2e9d5b", 5)):
        axis.set_yticks([])
        for token in table.itertuples() if not table.empty else []:
            start, end = float(token.start_sec), float(token.end_sec)
            axis.axvspan(start, end, color=color, alpha=.18)
            axis.axvline(start, color=color, lw=.6)
            axis.text((start + end) / 2, .5, str(token.label), ha="center", va="center",
                      fontsize=size, clip_on=True)
    axes[2].set_ylabel("words", rotation=0, ha="right")
    axes[3].set_ylabel("phones", rotation=0, ha="right")
    axes[3].set_xlim(0, duration)
    axes[3].set_xlabel("original recording time (s)")
    axes[0].set_title(title, fontsize=9, loc="left")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--screen-csv", required=True, help="sit_screen.csv")
    parser.add_argument("--json-dir", required=True, help="folder of <stem>.json transcripts")
    parser.add_argument("--audio-dir", required=True, help="folder of the SIT WAVs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--all-recordings", action="store_true",
                        help="also align recordings not marked looks_like_reading")
    parser.add_argument("--examiner-numbers", choices=["unk", "drop"], default="unk",
                        help="unk: model examiner item numbers as <unk> (default); "
                             "drop: leave them out of the text")
    parser.add_argument("--plots", choices=["flagged", "all", "none"], default="flagged")
    args = parser.parse_args(argv)
    table = run(screen_csv=args.screen_csv, json_dir=args.json_dir, audio_dir=args.audio_dir,
                out=args.out, profile_path=args.profile, usable_only=not args.all_recordings,
                examiner=args.examiner_numbers, plots=args.plots)
    return 0 if len(table) else 1


if __name__ == "__main__":
    sys.exit(main())
