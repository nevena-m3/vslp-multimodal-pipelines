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

READINGS_COLUMNS = ["file_name", "task_code", "task_id", "readings", "match_scores",
                    "transcript", "asr_model"]
MATCH_THRESHOLD = 0.6


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold().replace("-", " "))


def count_readings(transcript: str, prompt: str, threshold: float = MATCH_THRESHOLD) -> list[float]:
    """Return one similarity score per non-overlapping reading of ``prompt``.

    A reading is a window of transcript tokens whose token-level similarity
    to the prompt is at least ``threshold``; ASR slips ("hit me", "quay")
    still count, stray examiner speech does not.
    """
    said, target = tokens(transcript), tokens(prompt)
    if not said or not target:
        return []
    size = len(target)
    scores: list[float] = []
    index = 0
    while index < len(said):
        best, best_length = 0.0, 0
        for length in range(max(1, size - 2), size + 3):
            window = said[index:index + length]
            if not window:
                break
            ratio = SequenceMatcher(a=target, b=window, autojunk=False).ratio()
            if ratio > best:
                best, best_length = ratio, len(window)
        # The window must start on a prompt word, so it isn't shifted onto a
        # neighbouring reading.
        if best >= threshold and said[index] in target:
            scores.append(round(best, 3))
            index += best_length
        else:
            index += 1
    return scores


def _prompt_text(task_id: str) -> str:
    from .prompts import task_entry

    task = task_entry(task_id)
    if task is None or len(task.get("prompts", [])) != 1:
        raise ValueError(f"unknown_single_prompt_task:{task_id}")
    return str(task["prompts"][0]["exact_expected_text"])


def transcribe(paths: list[Path], model_name: str, model_dir: str | None, log=print) -> dict:
    from faster_whisper import WhisperModel  # only in the vslp-whisper environment

    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         download_root=model_dir)
    texts = {}
    for number, path in enumerate(paths, 1):
        segments, _info = model.transcribe(str(path), language="en", beam_size=5,
                                           vad_filter=False, condition_on_previous_text=False)
        texts[path.name] = " ".join(segment.text.strip() for segment in segments)
        log(f"{number}/{len(paths)} {path.name}")
    return texts


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
    rows: list[dict] = []
    if out.is_file():
        with out.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    done = {row["file_name"] for row in rows}
    todo = [item for item in wanted if item[0].name not in done]
    print(f"{len(wanted)} recordings need a reading count; {len(todo)} not yet done")
    texts = transcribe([item[0] for item in todo], args.model, args.model_dir) if todo else {}
    for path, code, task_id in todo:
        scores = count_readings(texts[path.name], _prompt_text(task_id))
        rows.append({"file_name": path.name, "task_code": code, "task_id": task_id,
                     "readings": len(scores), "match_scores": json.dumps(scores),
                     "transcript": texts[path.name], "asr_model": args.model})
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
