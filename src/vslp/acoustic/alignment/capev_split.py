"""Split 258_SENT_CAPEV recordings into their six CAPE-V sentences.

258_SENT_CAPEV holds six different sentences read once each in a fixed order
(confirmed on the 2026-11 speech-to-text screen):

    The blue spot is on the key again / How hard did he hit him? /
    We were away a year ago / We eat eggs every Easter. /
    My mama makes lemon jam / Peter will keep at the peak.

MFA aligns one known prompt per recording, so each recording is cut into six
sentence WAVs first. The cut never relies on speech-to-text timing alone:

1. Silero VAD finds speech chunks; Whisper (local) transcribes each chunk with
   word times.
2. The transcribed words are aligned to the 31 prompt words in order (global
   alignment with fuzzy word similarity), so every sentence gets its first and
   last matched word, even when Whisper mishears dysarthric speech.
3. Between two sentences, the cut goes in the middle of the longest silence
   (lowest-energy stretch) between the last word of one and the first word of
   the next. Cuts therefore fall in pauses, not in speech.
4. The sentence WAVs are exact sample copies of the original (native rate and
   format), named ``<original stem>_<S3|HIT|S5|EGGS|S6|S9>.wav`` so the batch
   aligner maps them to their registered prompt (``batch_tasks.yaml``). Their
   offset in the original recording is kept in ``capev_split.csv``.

A sentence that is not found, or two sentences with no pause between them, are
listed with a status and not written; they need a person.

Run in the ``vslp-whisper`` environment:
    python -m vslp.acoustic.alignment.capev_split --audio-dir W\\audio --out-dir W\\audio_capev
"""

from __future__ import annotations

import argparse
import csv
import re
import wave
from pathlib import Path

import numpy as np

from .readings import CHUNK_MIN_SILENCE_MS, _similar, load_16k, tokens

SOURCE_CODE = "258_SENT_CAPEV"
SENTENCES = [  # (file-name suffix, registered task_id, prompt text)
    ("S3", "sentence_s3", "The blue spot is on the key again"),
    ("HIT", "sentence_capev_hit", "How hard did he hit him?"),
    ("S5", "sentence_s5", "We were away a year ago"),
    ("EGGS", "sentence_capev_eggs", "We eat eggs every Easter."),
    ("S6", "sentence_s6", "My mama makes lemon jam"),
    ("S9", "sentence_s9", "Peter will keep at the peak."),
]
WORD_MATCH = 0.5        # word similarity that counts as the same word
MIN_FOUND_FRACTION = 0.5  # share of a sentence's words that must be matched
MIN_PAUSE_SEC = 0.08    # shorter gaps between sentences are "no pause"
OUTER_PAD_SEC = 0.5     # kept before the first and after the last sentence
SPLIT_COLUMNS = ["source_file", "split_file", "sentence", "task_id", "status",
                 "start_sec", "end_sec", "words_found", "words_expected", "pause_after_sec",
                 "transcript"]


def align_words(said: list[str], target: list[str]) -> list[int | None]:
    """For each target word, the index of the said word aligned to it (or None).

    Needleman-Wunsch: a pair of similar words scores its similarity, other
    pairs and gaps cost a little, so extra speech (repeats, comments) is
    skipped and misheard words still line up in order.
    """
    n, m = len(said), len(target)
    gap, mismatch = -0.2, -0.4
    score = np.zeros((n + 1, m + 1))
    score[1:, 0] = gap * np.arange(1, n + 1) * 0.5  # leading extra speech is cheap
    score[0, 1:] = gap * np.arange(1, m + 1)
    move = np.zeros((n + 1, m + 1), dtype=int)  # 0 diag, 1 skip said, 2 skip target
    move[1:, 0], move[0, 1:] = 1, 2
    sims = np.array([[_similar(a, b) for b in target] for a in said]) if n and m else None
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = sims[i - 1, j - 1]
            options = (score[i - 1, j - 1] + (sim if sim >= WORD_MATCH else mismatch),
                       score[i - 1, j] + (gap * 0.5 if j in (0, m) else gap),
                       score[i, j - 1] + gap)
            move[i, j] = int(np.argmax(options))
            score[i, j] = options[move[i, j]]
    result: list[int | None] = [None] * m
    i, j = n, m
    while i > 0 and j > 0:
        if move[i, j] == 0:
            if sims[i - 1, j - 1] >= WORD_MATCH:
                result[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif move[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return result


def _quietest_point(energy_db: np.ndarray, hop: float, start: float, end: float) -> tuple[float, float]:
    """Middle of the longest stretch within 6 dB of the minimum between start and end.

    Returns (cut time, length of that quiet stretch).
    """
    a, b = int(start / hop), max(int(start / hop) + 1, int(np.ceil(end / hop)))
    segment = energy_db[a:b]
    if len(segment) == 0:
        return (start + end) / 2, 0.0
    quiet = segment <= float(np.min(segment)) + 6.0
    best, best_length, run_start = (0, 0), 0, None
    for index, flag in enumerate(np.r_[quiet, False]):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            if index - run_start > best_length:
                best, best_length = (run_start, index), index - run_start
            run_start = None
    middle = (best[0] + best[1]) / 2
    return (a + middle) * hop, best_length * hop


def plan_split(words: list[dict], energy_db: np.ndarray, hop: float, duration: float) -> list[dict]:
    """Six sentence spans from timed words (``{"word", "start", "end"}``)."""
    said = [token for word in words for token in tokens(word["word"])]
    owner = [index for index, word in enumerate(words) for _ in tokens(word["word"])]
    target, sentence_of = [], []
    for number, (_suffix, _task, text) in enumerate(SENTENCES):
        for token in tokens(text):
            target.append(token)
            sentence_of.append(number)
    matched = align_words(said, target)
    spans = []
    for number, (suffix, task_id, text) in enumerate(SENTENCES):
        hits = [matched[j] for j in range(len(target)) if sentence_of[j] == number
                and matched[j] is not None]
        expected = sum(1 for owner_sentence in sentence_of if owner_sentence == number)
        found = len(hits) >= MIN_FOUND_FRACTION * expected
        spans.append({
            "sentence": suffix, "task_id": task_id, "words_found": len(hits),
            "words_expected": expected, "found": found,
            "first": float(words[owner[min(hits)]]["start"]) if found else None,
            "last": float(words[owner[max(hits)]]["end"]) if found else None})
    # Cuts between consecutive found sentences, in the quietest stretch.
    found = [span for span in spans if span["found"]]
    for left, right in zip(found, found[1:]):
        if right["first"] - left["last"] < MIN_PAUSE_SEC:
            left["pause_after"] = max(0.0, right["first"] - left["last"])
            cut = (left["last"] + right["first"]) / 2
        else:
            cut, length = _quietest_point(energy_db, hop, left["last"], right["first"])
            left["pause_after"] = length
        left["cut_after"] = right["cut_before"] = cut
    for span in spans:
        if not span["found"]:
            span["status"] = "SENTENCE_NOT_FOUND"
            continue
        span["start"] = span.get("cut_before", max(0.0, span["first"] - OUTER_PAD_SEC))
        span["end"] = span.get("cut_after", min(duration, span["last"] + OUTER_PAD_SEC))
        span["status"] = "OK"
    for left, right in zip(found, found[1:]):
        if left.get("pause_after", 1.0) < MIN_PAUSE_SEC:
            left["status"] = right["status"] = "NO_PAUSE_BETWEEN_SENTENCES"
    return spans


def write_slice(source: Path, target: Path, start: float, end: float) -> None:
    """Exact copy of [start, end) of a PCM WAV, same rate, width and channels."""
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        rate = reader.getframerate()
        first = max(0, int(round(start * rate)))
        last = min(reader.getnframes(), int(round(end * rate)))
        reader.setpos(first)
        frames = reader.readframes(last - first)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


def energy_contour(audio16k: np.ndarray, hop: float = 0.01) -> np.ndarray:
    step = int(16000 * hop)
    frames = len(audio16k) // step
    power = np.square(audio16k[:frames * step].astype(np.float64)).reshape(frames, step).mean(axis=1)
    return 10 * np.log10(np.maximum(power, 1e-12))


def transcribe_words(paths: list[Path], model_name: str, model_dir: str | None, log=print):
    """Yield (path, 16 kHz audio, timed words) per file; chunk offsets added back."""
    from faster_whisper import WhisperModel  # only in the vslp-whisper environment
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=model_dir)
    options = VadOptions(min_silence_duration_ms=CHUNK_MIN_SILENCE_MS, speech_pad_ms=150)
    for number, path in enumerate(paths, 1):
        audio = load_16k(path)
        words = []
        for stamp in get_speech_timestamps(audio, options):
            piece = audio[stamp["start"]:stamp["end"]]
            if len(piece) < 0.25 * 16000:
                continue
            offset = stamp["start"] / 16000
            segments, _info = model.transcribe(piece, language="en", beam_size=5,
                                               vad_filter=False, word_timestamps=True,
                                               condition_on_previous_text=False)
            for segment in segments:
                for word in segment.words or []:
                    words.append({"word": word.word, "start": offset + word.start,
                                  "end": offset + word.end})
        log(f"{number}/{len(paths)} {path.name}: {len(words)} words")
        yield path, audio, words


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split 258_SENT_CAPEV into six sentence WAVs")
    parser.add_argument("--audio-dir", required=True, help="folder with the original WAVs")
    parser.add_argument("--out-dir", required=True, help="folder for the sentence WAVs")
    parser.add_argument("--model", default="medium.en")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    pattern = re.compile(rf"_{SOURCE_CODE}$")
    sources = sorted(path for path in Path(args.audio_dir).rglob("*.wav") if pattern.search(path.stem))
    table_path = out_dir / "capev_split.csv"
    rows: list[dict] = []
    if table_path.is_file():
        with table_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    done = {row["source_file"] for row in rows}
    todo = [path for path in sources if path.name not in done]
    print(f"{len(sources)} {SOURCE_CODE} recordings; {len(todo)} not yet split")
    for source, audio, words in transcribe_words(todo, args.model, args.model_dir):
        name = source.name
        hop = 0.01
        spans = plan_split(words, energy_contour(audio, hop), hop, len(audio) / 16000)
        text = " ".join(word["word"].strip() for word in words)
        for span in spans:
            split_name = ""
            if span["status"] == "OK":
                split_name = f"{source.stem}_{span['sentence']}.wav"
                write_slice(source, out_dir / split_name, span["start"], span["end"])
            rows.append({"source_file": name, "split_file": split_name,
                         "sentence": span["sentence"], "task_id": span["task_id"],
                         "status": span["status"],
                         "start_sec": round(span.get("start", float("nan")), 3),
                         "end_sec": round(span.get("end", float("nan")), 3),
                         "words_found": span["words_found"],
                         "words_expected": span["words_expected"],
                         "pause_after_sec": round(span.get("pause_after", float("nan")), 3),
                         "transcript": text})
        out_dir.mkdir(parents=True, exist_ok=True)
        with table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print(f"Wrote {table_path}: {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
