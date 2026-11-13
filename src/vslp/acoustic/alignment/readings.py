"""Count how many times a known prompt was read in each recording.

Repetition counts differ by protocol and speaker (e.g. "Buy Bobby a puppy" is
read 3 to ~8 times), and MFA must be given exactly the number of readings that
were spoken. Speech-to-text (faster-whisper, run locally) is used ONLY to count
readings of the known text; MFA still aligns the canonical prompt, never the
ASR words.

Run in the separate ``vslp-whisper`` environment (faster-whisper installed)::

    python -m vslp.acoustic.alignment.readings --audio-dir W\\audio \
        --out W\\readings.csv --model medium.en --model-dir mfa_batch\\models

Then pass ``--readings W\\readings.csv`` to the batch aligner.
"""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

# Kept import-light on purpose: this runs in the small vslp-whisper environment
# (faster-whisper + pyyaml), not in the GUI environment.
NAME_PATTERN = re.compile(r"^[^_]+_\d+_\d+_\d{8}_(?P<code>.+)$")
TASKS_PATH = Path(__file__).with_name("batch_tasks.yaml")

READINGS_COLUMNS = ["file_name", "task_code", "task_id", "readings", "count_source",
                    "match_scores", "speech_chunks", "speech_sec", "transcript", "asr_model"]
MATCH_THRESHOLD = 0.6
# Speech chunks are transcribed one by one: Whisper on a whole recording tends
# to write a repeated sentence only once (2026-11 pilot 2), which undercounts.
CHUNK_MIN_SILENCE_MS = 300


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold().replace("-", " "))


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def count_readings(transcript: str, prompt: str, threshold: float = MATCH_THRESHOLD) -> list[float]:
    """Return one similarity score per non-overlapping reading of ``prompt``.

    Matching is on characters over a window of words, so ASR slips on
    dysarthric speech ("bye bubby and puppy", "hit me") still count, while
    stray speech ("thank you", instructions) does not. A reading must start
    on a word that resembles one of the prompt's first two words, so a false
    start ("did he hit him?") is not counted as a reading.
    """
    said, target = tokens(transcript), tokens(prompt)
    if not said or not target:
        return []
    size, prompt_text = len(target), " ".join(target)
    scores: list[float] = []
    index = 0
    while index < len(said):
        best, best_length = 0.0, 0
        for length in range(max(1, size - 2), size + 3):
            window = said[index:index + length]
            if len(window) < length and length > size:
                break
            ratio = _similar(prompt_text, " ".join(window))
            if ratio > best:
                best, best_length = ratio, len(window)
        starts_like_prompt = max(_similar(said[index], word) for word in target[:2]) >= .5
        if best >= threshold and starts_like_prompt:
            scores.append(round(best, 3))
            index += best_length
        else:
            index += 1
    return scores


def estimate_readings(chunks: list[dict], prompt: str) -> tuple[int, list[float], str]:
    """Readings in one recording from its per-chunk transcripts.

    Readings are counted in the joined chunk texts, so a reading split across
    two chunks ("my mama | makes lemon jam") still counts once.

    Chunk duration is deliberately NOT used to add readings. In pilot 4 a
    duration rule overcounted clean 3-reading files (a chunk that also held a
    pause or a breath looked like two readings), and MFA then squeezed the
    extra copy into a pause: a false AUTO_PASS. An undercount is safe by
    comparison, because structural QC sends it to review (long word spans).
    """
    joined = " | ".join(chunk["text"] for chunk in chunks if chunk["text"])
    scores = count_readings(joined, prompt)
    return len(scores), scores, "chunks"


def _prompt_text(task_id: str) -> str:
    from .prompts import task_entry

    task = task_entry(task_id)
    if task is None or len(task.get("prompts", [])) != 1:
        raise ValueError(f"unknown_single_prompt_task:{task_id}")
    return str(task["prompts"][0]["exact_expected_text"])


def load_16k(path: Path):
    """PCM WAV -> mono float32 at 16 kHz with numpy only (no PyAV version issues)."""
    import wave

    import numpy as np

    try:
        with wave.open(str(path), "rb") as handle:
            rate, width, channels = (handle.getframerate(), handle.getsampwidth(),
                                     handle.getnchannels())
            raw = handle.readframes(handle.getnframes())
    except wave.Error:  # e.g. float WAV: fall back to faster-whisper's decoder
        from faster_whisper.audio import decode_audio

        return decode_audio(str(path), sampling_rate=16000)
    if width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data = (bytes_[:, 0].astype(np.int32) | bytes_[:, 1].astype(np.int32) << 8 |
                bytes_[:, 2].astype(np.int32) << 16)
        data = np.where(data >= 1 << 23, data - (1 << 24), data) / float(1 << 23)
    else:
        kind = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        data = np.frombuffer(raw, dtype=kind).astype(np.float64)
        data = (data - 128) / 128 if width == 1 else data / float(np.iinfo(kind).max)
    audio = data.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        # Windowed-sinc low-pass at 7.6 kHz, then linear resampling.
        taps = np.arange(-64, 65)
        cutoff = 7600 / rate
        kernel = 2 * cutoff * np.sinc(2 * cutoff * taps) * np.hamming(len(taps))
        audio = np.convolve(audio, kernel / kernel.sum(), mode="same")
        times = np.arange(0, len(audio) / rate, 1 / 16000)
        audio = np.interp(times, np.arange(len(audio)) / rate, audio)
    return audio.astype(np.float32)


def transcribe(paths: list[Path], model_name: str, model_dir: str | None, log=print,
               prompts: dict | None = None, on_result=None) -> dict:
    """Transcribe each speech chunk separately (and the whole file as a fallback)."""
    from faster_whisper import WhisperModel  # only in the vslp-whisper environment
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         download_root=model_dir)
    options = VadOptions(min_silence_duration_ms=CHUNK_MIN_SILENCE_MS, speech_pad_ms=150)

    def text_of(audio) -> str:
        segments, _info = model.transcribe(audio, language="en", beam_size=5,
                                           vad_filter=False, condition_on_previous_text=False)
        return " ".join(segment.text.strip() for segment in segments).strip()

    results = {}
    for number, path in enumerate(paths, 1):
        audio = load_16k(path)
        chunks = []
        for stamp in get_speech_timestamps(audio, options):
            piece = audio[stamp["start"]:stamp["end"]]
            if len(piece) >= 0.25 * 16000:
                chunks.append({"start": stamp["start"] / 16000, "end": stamp["end"] / 16000,
                               "text": text_of(piece)})
        result = {"chunks": chunks, "whole_text": ""}
        prompt = (prompts or {}).get(path.name)
        # Very short chunks of dysarthric speech can transcribe as nonsense;
        # if no reading is found in them, try the whole file once (pilot 3).
        if prompt and chunks and estimate_readings(chunks, prompt)[0] == 0:
            result["whole_text"] = text_of(audio)
        results[path.name] = result
        if on_result is not None:
            on_result(path, result)
        log(f"{number}/{len(paths)} {path.name}: {len(chunks)} speech chunks")
    return results


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description="Count prompt readings per recording")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="medium.en")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args(argv)
    table = yaml.safe_load(TASKS_PATH.read_text(encoding="utf-8"))["tasks"]
    wanted = []
    for path in sorted(Path(args.audio_dir).rglob("*.wav")):
        match = NAME_PATTERN.match(path.stem)
        spec = table.get(match["code"]) if match else None
        if spec and spec.get("count_readings"):
            wanted.append((path, match["code"], spec["task_id"]))
    out = Path(args.out)
    # Whisper output per file is cached as it is produced (one JSON line per
    # file), so a long run can resume after an interruption and every count
    # can be recomputed offline when the counting rule changes.
    cache_path = Path(str(out) + ".chunks.jsonl")
    cache: dict[str, dict] = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                cache[item["file_name"]] = {"chunks": item["chunks"],
                                            "whole_text": item.get("whole_text", "")}
    todo = [item for item in wanted if item[0].name not in cache]
    print(f"{len(wanted)} recordings need a reading count; {len(todo)} not yet transcribed")
    prompts = {item[0].name: _prompt_text(item[2]) for item in wanted}

    def remember(path: Path, result: dict) -> None:
        cache[path.name] = result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"file_name": path.name, **result}) + "\n")

    if todo:
        transcribe([item[0] for item in todo], args.model, args.model_dir,
                   prompts=prompts, on_result=remember)
    rows: list[dict] = []
    for path, code, task_id in wanted:
        result = cache[path.name]
        prompt = prompts[path.name]
        count, scores, source = estimate_readings(result["chunks"], prompt)
        if count == 0 and result["whole_text"]:
            scores = count_readings(result["whole_text"], prompt)
            count, source = len(scores), "whole_file" if scores else "none"
        rows.append({"file_name": path.name, "task_code": code, "task_id": task_id,
                     "readings": count, "count_source": source if count else "none",
                     "match_scores": json.dumps(scores),
                     "speech_chunks": len(result["chunks"]),
                     "speech_sec": round(sum(c["end"] - c["start"] for c in result["chunks"]), 2),
                     "transcript": " | ".join(c["text"] for c in result["chunks"])
                     + (f" || whole: {result['whole_text']}" if result["whole_text"] else ""),
                     "asr_model": args.model})
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=READINGS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[tuple, int] = {}
    for row in rows:
        key = (row["task_code"], str(row["readings"]))
        summary[key] = summary.get(key, 0) + 1
    for (code, count), number in sorted(summary.items()):
        print(f"{code}: {count} readings -> {number} recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
