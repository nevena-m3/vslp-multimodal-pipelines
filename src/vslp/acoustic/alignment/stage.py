"""Versioned linguistic alignment on frozen, reviewed patient-task audio.

All public token times are on the original recording clock. The stage never
changes canonical audio or frozen segmentation. Imported annotation and MFA
provider output pass through the same validation and final schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import unicodedata
from typing import Callable, Protocol
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import soundfile as sf

from vslp.core.schemas import StageResult
from .mfa_provider import MfaProfile, MfaProvider as MFAProvider, parse_textgrid as _parse_textgrid


ALIGNMENT_VERSION = "reviewed-alignment-1.0.0"
PHONE_SET = "ARPABET_CMU_39"
ARPABET_VOWELS = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
})
ARPABET_CONSONANTS = frozenset({
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N",
    "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
})
WORD_COLUMNS = (
    "alignment_run_id", "recording_id", "file_name", "task_id", "prompt_version",
    "word_index", "word", "start_sec", "end_sec", "duration_sec", "start_sample",
    "end_sample", "sequence_id", "alignment_status", "alignment_source",
)
PHONE_COLUMNS = (
    "alignment_run_id", "recording_id", "file_name", "task_id", "prompt_version",
    "word_index", "phone_index", "phone", "phone_raw", "phone_normalized",
    "stress", "is_vowel", "start_sec", "end_sec", "duration_sec", "start_sample",
    "end_sample", "sequence_id", "alignment_status", "alignment_source",
)
DIAGNOSTIC_COLUMNS = (
    "alignment_run_id", "recording_id", "file_name", "status", "reason",
    "expected_word_count", "aligned_expected_word_count", "word_coverage",
    "missing_words", "extra_tokens", "n_words", "n_phones", "invalid_intervals",
    "overlapping_intervals", "nonmonotonic_intervals", "outside_domain_tokens",
    "canonical_sample_rate_hz", "working_sample_rate_hz",
)


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            table.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class PromptManifest:
    manifest_version: str
    task_id: str
    prompt_version: str
    language: str
    transcript: str
    expected_words: tuple[str, ...]
    phone_set: str

    @classmethod
    def load(cls, path: str | Path) -> "PromptManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        required = ("manifest_version", "task_id", "prompt_version", "language",
                    "transcript", "expected_words", "phone_set")
        if any(key not in raw for key in required):
            raise ValueError("missing_prompt_fields")
        if not isinstance(raw["expected_words"], list) or not raw["expected_words"]:
            raise ValueError("missing_expected_words")
        if any(not str(raw[key]).strip() for key in required if key != "expected_words"):
            raise ValueError("missing_prompt_fields")
        if raw["phone_set"] != PHONE_SET:
            raise ValueError("unsupported_phone_set")
        # Match word boundaries without imposing MFA's separate spoken-number gate:
        # imported validated Alignment manifests can use explicit numbered labels.
        # Hyphenated orthography still resolves to the two expected MFA words.
        transcript_words = re.findall(r"[\w']+", str(raw["transcript"]).casefold())
        expected = re.findall(r"[\w']+", " ".join(
            str(word) for word in raw["expected_words"]).casefold())
        if transcript_words != expected:
            raise ValueError("PROMPT_MISMATCH")
        return cls(*(tuple(str(word) for word in raw[key]) if key == "expected_words"
                     else str(raw[key]) for key in required))


@dataclass(frozen=True)
class AlignmentConfig:
    source: str = "external"  # external or mfa
    prompt_manifest_path: str = ""
    words_csv: str = ""
    phones_csv: str = ""
    provider: "AlignmentProvider | None" = None
    acoustic_model: str = ""
    acoustic_model_version: str = ""
    dictionary: str = ""
    dictionary_version: str = ""
    working_sample_rate_hz: int = 16000
    mfa_profile_path: str = ""
    speaker_manifest_path: str = ""
    transcript_overrides_path: str = ""
    recording_prompt_manifest_paths: dict[str, str] | None = None
    trial_manifest_path: str = ""
    rerun_recording_id: str = ""
    reuse_run_id: str = ""
    structural_auto_review: bool = False


def _trial_units(config: AlignmentConfig, kept: pd.DataFrame) -> dict[str, list[dict]]:
    """Explicit, reviewed trial intervals; no speech-gap or filename inference."""
    if not config.trial_manifest_path:
        return {str(row.recording_id): [{"trial_id": "legacy_whole_recording",
                 "start_sec": float(row.analysis_start_sec),
                 "end_sec": float(row.analysis_end_sec), "prompt_id": ""}]
                for row in kept.itertuples()}
    raw = json.loads(Path(config.trial_manifest_path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1":
        raise ValueError("invalid_trial_manifest_version")
    grouped: dict[str, list[dict]] = {str(value): [] for value in kept.recording_id}
    by_record = {str(row.recording_id): row for row in kept.itertuples()}
    for item in raw.get("trials", []):
        identity = str(item.get("recording_id", ""))
        trial_id = str(item.get("trial_id", ""))
        if identity not in by_record or not re.fullmatch(r"[A-Za-z0-9_]+", trial_id):
            raise ValueError("invalid_trial_identity")
        start, end = float(item["start_sec"]), float(item["end_sec"])
        record = by_record[identity]
        if not (np.isfinite([start, end]).all() and
                float(record.analysis_start_sec) <= start < end <= float(record.analysis_end_sec)):
            raise ValueError("trial_outside_reviewed_analysis")
        grouped[identity].append({"trial_id": trial_id, "start_sec": start,
                                  "end_sec": end, "prompt_id": str(item.get("prompt_id", ""))})
    for identity, items in grouped.items():
        if not items:
            raise ValueError(f"trial_definition_required:{identity}")
        items.sort(key=lambda value: value["start_sec"])
        if len({item["trial_id"] for item in items}) != len(items) or any(
                right["start_sec"] < left["end_sec"] - 1e-7
                for left, right in zip(items, items[1:])):
            raise ValueError("overlapping_or_duplicate_trials")
    return grouped


@dataclass(frozen=True)
class TimePiece:
    working_start_sec: float
    working_end_sec: float
    original_start_sec: float
    original_end_sec: float
    sequence_id: int


def map_interval(start: float, end: float, pieces: list[TimePiece]) -> tuple[float, float, int]:
    """Reject tokens crossing a removed contamination gap."""
    for piece in pieces:
        if start >= piece.working_start_sec - 1e-7 and end <= piece.working_end_sec + 1e-7:
            scale = ((piece.original_end_sec - piece.original_start_sec) /
                     (piece.working_end_sec - piece.working_start_sec))
            return (piece.original_start_sec + (start - piece.working_start_sec) * scale,
                    piece.original_start_sec + (end - piece.working_start_sec) * scale,
                    piece.sequence_id)
    raise ValueError("token_crosses_excluded_gap")


def _analysis_pieces(intervals: pd.DataFrame, start: float, end: float) -> list[tuple[float, float]]:
    exclusions = sorted((max(start, float(row.start_sec)), min(end, float(row.end_sec)))
                        for row in intervals.itertuples() if row.segment_role == "manual_exclusion"
                        and float(row.end_sec) > start and float(row.start_sec) < end)
    pieces = []
    cursor = start
    for left, right in exclusions:
        if left > cursor:
            pieces.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def _working_audio(path: Path, spans: list[tuple[float, float]], rate: int,
                   destination: Path) -> tuple[list[TimePiece], int]:
    audio, native_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    fraction = Fraction(rate, native_rate).limit_denominator()
    blocks = []
    mapping = []
    offset_samples = 0
    for index, (start, end) in enumerate(spans, 1):
        first_sample, last_sample = round(start * native_rate), round(end * native_rate)
        block = audio[first_sample:last_sample]
        if rate != native_rate:
            block = resample_poly(block, fraction.numerator, fraction.denominator).astype("float32")
        if len(block) == 0:
            continue
        mapping.append(TimePiece(offset_samples / rate, (offset_samples + len(block)) / rate,
                                 first_sample / native_rate, last_sample / native_rate, index))
        offset_samples += len(block)
        blocks.append(block)
    if not blocks:
        raise ValueError("empty_reviewed_analysis_domain")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.concatenate(blocks), rate, subtype="FLOAT")
    return mapping, native_rate


class AlignmentProvider(Protocol):
    name: str
    version: str

    def align_corpus(self, corpus: Path, output: Path, profile: MfaProfile | None,
                     logs: Path) -> dict[str, Path]: ...


def normalize_transcript(text: str) -> str:
    """vslp_prompt_nfkc_v1: uppercase, preserve apostrophes, split hyphens.

    Digits and abbreviations require a manually reviewed spoken form; no
    implicit number expansion or ASR transcription is performed.
    """
    value = unicodedata.normalize("NFKC", text).replace("’", "'").replace("-", " ")
    value = re.sub(r"[^\w'\s]", " ", value, flags=re.UNICODE)
    value = " ".join(value.upper().split())
    if any(char.isdigit() for char in value):
        raise ValueError("TRANSCRIPT_NUMBER_REQUIRES_REVIEW")
    return value


def _transcripts(prompt: PromptManifest, config: AlignmentConfig, records: pd.DataFrame,
                 record_prompts: dict[str, PromptManifest],
                 destination: Path) -> dict[str, str]:
    overrides = {}
    if config.transcript_overrides_path:
        raw = json.loads(Path(config.transcript_overrides_path).read_text(encoding="utf-8"))
        if raw.get("prompt_version") != prompt.prompt_version:
            raise ValueError("PROMPT_MISMATCH")
        for item in raw.get("overrides", []):
            if not all(str(item.get(key, "")).strip() for key in
                       ("recording_id", "alignment_transcript", "reason", "reviewer", "reviewed_utc")):
                raise ValueError("invalid_transcript_override")
            identity = str(item["recording_id"])
            if identity in overrides:
                raise ValueError("duplicate_transcript_override")
            if (identity in record_prompts and item.get("prompt_version")
                    and item["prompt_version"] != record_prompts[identity].prompt_version):
                raise ValueError("PROMPT_MISMATCH")
            overrides[identity] = item["alignment_transcript"]
    identities = set(records.recording_id.astype(str))
    if set(overrides) - identities:
        raise ValueError("unknown_transcript_override_recording")
    resolved = {identity: normalize_transcript(overrides.get(
        identity, record_prompts[identity].transcript))
                for identity in identities}
    _atomic_json({"normalization_profile_id": "vslp_prompt_nfkc_v1",
                  "expected_prompt": prompt.transcript,
                  "prompt_version": prompt.prompt_version,
                  "expected_prompt_by_recording": {
                      identity: item.transcript for identity, item in record_prompts.items()},
                  "expected_transcript_sha256_by_recording": {
                      identity: sha256(item.transcript.encode("utf-8")).hexdigest()
                      for identity, item in record_prompts.items()},
                  "transcripts": resolved,
                  "alignment_transcript_sha256_by_recording": {
                      identity: sha256(text.encode("utf-8")).hexdigest()
                      for identity, text in resolved.items()},
                  "overrides_source_sha256": (_hash(Path(config.transcript_overrides_path))
                                              if config.transcript_overrides_path else "")}, destination)
    return resolved


def _speakers(config: AlignmentConfig, records: pd.DataFrame) -> dict[str, str]:
    if not config.speaker_manifest_path:
        raise ValueError("MISSING_SPEAKER_MAPPING")
    raw = json.loads(Path(config.speaker_manifest_path).read_text(encoding="utf-8"))
    mapping = {str(item["recording_id"]): str(item["speaker_id"])
               for item in raw.get("recordings", [])}
    if len(mapping) != len(raw.get("recordings", [])):
        raise ValueError("duplicate_speaker_mapping")
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", value) for value in mapping.values()):
        raise ValueError("invalid_speaker_id")
    if set(records.recording_id.astype(str)) - set(mapping):
        raise ValueError("MISSING_SPEAKER_MAPPING")
    return mapping


def normalize_phone(label: str, phone_set: str = PHONE_SET) -> tuple[str, str, bool]:
    if phone_set != PHONE_SET:
        raise ValueError("unsupported_phone_set")
    raw = str(label).strip().upper()
    match = re.fullmatch(r"([A-Z]+)([012])?", raw)
    if not match:
        raise ValueError("invalid_phone_label")
    normalized, stress = match.group(1), match.group(2) or ""
    if normalized not in ARPABET_VOWELS | ARPABET_CONSONANTS:
        raise ValueError("PHONESET_MAPPING_REQUIRED")
    return normalized, stress, normalized in ARPABET_VOWELS


def _source_rows(path: str, token_type: str) -> pd.DataFrame:
    if not path or not Path(path).is_file():
        return pd.DataFrame()
    table = pd.read_csv(path, keep_default_na=False)
    required = {"recording_id", "start_sec", "end_sec"}
    if not required <= set(table):
        raise ValueError("alignment_import_schema_missing")
    if not ({"label", "word"} & set(table)) and token_type == "word":
        raise ValueError("alignment_import_word_label_missing")
    if not ({"label", "phone"} & set(table)) and token_type == "phone":
        raise ValueError("alignment_import_phone_label_missing")
    if "token_type" in table:
        table = table.loc[table.token_type.astype(str).str.lower().eq(token_type)].copy()
    return table


def _token_rows(raw: pd.DataFrame, *, kind: str, record: object, prompt: PromptManifest,
                run_id: str, source: str, native_rate: int, pieces: list[TimePiece] | None,
                domain: list[tuple[float, float]]) -> tuple[list[dict], int, int]:
    rows = []
    invalid = 0
    outside = 0
    for index, item in enumerate(raw.to_dict("records"), 1):
        try:
            if "recording_id" in item and str(item["recording_id"]) != str(record.recording_id):
                continue
            start, end = float(item["start_sec"]), float(item["end_sec"])
            if not np.isfinite([start, end]).all() or end <= start:
                raise ValueError("invalid_alignment_boundary")
            sequence = 1
            if pieces is not None:
                start, end, sequence = map_interval(start, end, pieces)
            else:
                matches = [i for i, (left, right) in enumerate(domain, 1)
                           if left - 1e-7 <= start < end <= right + 1e-7]
                if not matches:
                    raise ValueError("token_outside_reviewed_domain")
                sequence = matches[0]
            label = str(item.get("word" if kind == "word" else "phone",
                                 item.get("label", ""))).strip()
            if not label:
                raise ValueError("empty_token_label")
            common = {"alignment_run_id": run_id, "recording_id": str(record.recording_id),
                      "file_name": str(record.file_name), "task_id": prompt.task_id,
                      "prompt_version": prompt.prompt_version, "start_sec": start,
                      "end_sec": end, "duration_sec": end - start,
                      "start_sample": round(start * native_rate),
                      "end_sample": round(end * native_rate), "sequence_id": sequence,
                      "alignment_status": "ALIGNED", "alignment_source": source}
            if kind == "word":
                common.update({"word_index": int(item.get("word_index") or index), "word": label})
            else:
                normalized, stress, vowel = normalize_phone(label, prompt.phone_set)
                common.update({"word_index": int(item.get("word_index") or 0),
                               "phone_index": int(item.get("phone_index") or index),
                               "phone": label, "phone_raw": label,
                               "phone_normalized": normalized, "stress": stress,
                               "is_vowel": vowel})
            rows.append(common)
        except (KeyError, TypeError, ValueError) as exc:
            invalid += 1
            if str(exc) in {"token_outside_reviewed_domain", "token_crosses_excluded_gap"}:
                outside += 1
    return rows, invalid, outside


def _coverage(words: list[dict], expected: tuple[str, ...]) -> tuple[int, list[str], int]:
    # Order-preserving dynamic program handles omissions and repeated words.
    aligned = [re.sub(r"[^\w']", "", row["word"]).casefold() for row in words]
    target = [re.sub(r"[^\w']", "", word).casefold() for word in expected]
    lengths = [[0] * (len(aligned) + 1) for _ in range(len(target) + 1)]
    for i in range(len(target) - 1, -1, -1):
        for j in range(len(aligned) - 1, -1, -1):
            lengths[i][j] = (1 + lengths[i + 1][j + 1] if target[i] == aligned[j]
                             else max(lengths[i + 1][j], lengths[i][j + 1]))
    i = j = 0
    missing = []
    while i < len(target):
        if j < len(aligned) and target[i] == aligned[j]:
            i += 1
            j += 1
        elif j < len(aligned) and lengths[i][j + 1] >= lengths[i + 1][j]:
            j += 1
        else:
            missing.append(target[i])
            i += 1
    matched = len(target) - len(missing)
    return matched, missing, max(0, len(aligned) - matched)


def _validate_order(rows: list[dict]) -> tuple[int, int]:
    ordered = sorted(rows, key=lambda item: item["start_sec"])
    overlaps = sum(float(right["start_sec"]) < float(left["end_sec"]) - 1e-7
                   for left, right in zip(ordered, ordered[1:]))
    nonmonotonic = sum(float(right["start_sec"]) < float(left["start_sec"])
                       for left, right in zip(rows, rows[1:]))
    return overlaps, nonmonotonic


def _paths(root: Path) -> tuple[Path, Path]:
    base = root / "acoustic" / "004_alignment"
    return base, base / "final"


def list_alignment_runs(output_root: str | Path) -> list[dict]:
    registry = _paths(Path(output_root))[0] / "logs" / "alignment_registry.json"
    return json.loads(registry.read_text(encoding="utf-8")).get("runs", []) if registry.is_file() else []


def run_acoustic_alignment(output_root: str | Path, config: AlignmentConfig,
                           progress_callback: Callable[[int, int, str], None] | None = None) -> StageResult:
    """Create a historical alignment run; explicit freeze publishes the final product."""
    root = Path(output_root)
    base, _ = _paths(root)
    review = root / "acoustic" / "003_segmentation_review" / "final"
    decision_path = review / "final_segmentation_decisions.csv"
    interval_path = review / "final_segmentation_intervals.csv"
    if not decision_path.is_file() or not interval_path.is_file():
        raise FileNotFoundError("missing_final_reviewed_segmentation")
    if not config.prompt_manifest_path:
        raise ValueError("MISSING_PROMPT")
    prompt = PromptManifest.load(config.prompt_manifest_path)
    run_meta = json.loads((root / "project_manifest.json").read_text(encoding="utf-8"))
    if prompt.task_id != str(run_meta.get("task_id", "")):
        raise ValueError("PROMPT_MISMATCH")
    if config.source not in {"external", "mfa"}:
        raise ValueError("unsupported_alignment_source")
    if config.source == "external" and not config.words_csv and not config.phones_csv:
        raise ValueError("missing_validated_alignment_tables")
    if config.source == "mfa" and config.provider is None and not config.mfa_profile_path:
        raise ValueError("MISSING_MFA_PROFILE")
    decisions = pd.read_csv(decision_path, keep_default_na=False)
    intervals = pd.read_csv(interval_path, keep_default_na=False)
    kept = decisions.loc[decisions.final_decision.isin(["KEEP_AUTO", "KEEP_MANUAL"])]
    if kept.empty:
        raise ValueError("no_kept_recordings_to_align")
    trials_by_record = _trial_units(config, kept)
    prior_run = None
    prior_manifest: dict = {}
    prior_tables: dict[str, pd.DataFrame] = {}
    if config.rerun_recording_id:
        if not config.reuse_run_id or config.rerun_recording_id not in set(
                kept.recording_id.astype(str)):
            raise ValueError("invalid_targeted_alignment_rerun")
        prior_run = base / "runs" / config.reuse_run_id
        prior_manifest = json.loads((prior_run / "logs" / "stage_manifest.json").read_text(
            encoding="utf-8"))
        if (prior_manifest["source_final_decisions_sha256"] != _hash(decision_path)
                or prior_manifest["source_final_intervals_sha256"] != _hash(interval_path)):
            raise ValueError("reviewed_segmentation_changed_since_alignment")
        old_trials = json.loads((prior_run / "configs" / "alignment_trials.json").read_text(
            encoding="utf-8"))["trials"]
        for identity, current in trials_by_record.items():
            if identity == config.rerun_recording_id:
                continue
            previous = sorted((item for item in old_trials if str(item["recording_id"]) == identity),
                              key=lambda item: float(item["start_sec"]))
            if current != [{key: item[key] for key in ("trial_id", "start_sec", "end_sec", "prompt_id")}
                           for item in previous]:
                raise ValueError("unaffected_trial_plan_changed")
        for name in ("alignment_words", "alignment_phones", "alignment_diagnostics",
                     "alignment_trial_diagnostics", "alignment_trial_tokens"):
            prior_tables[name] = pd.read_csv(prior_run / "tables" / f"{name}.csv",
                                             keep_default_na=False)
    record_prompts = {str(identity): prompt for identity in kept.recording_id.astype(str)}
    if config.recording_prompt_manifest_paths:
        for identity, source in config.recording_prompt_manifest_paths.items():
            if identity not in record_prompts:
                raise ValueError("unknown_recording_prompt_mapping")
            individual = PromptManifest.load(source)
            if individual.task_id != prompt.task_id or individual.language != prompt.language:
                raise ValueError("PROMPT_MISMATCH")
            record_prompts[identity] = individual
    if config.rerun_recording_id:
        if (prior_manifest.get("mfa_profile_sha256") and config.mfa_profile_path
                and prior_manifest["mfa_profile_sha256"] != _hash(Path(config.mfa_profile_path))):
            raise ValueError("alignment_profile_changed_since_prior_run")
        old_prompts_path = prior_run / "configs" / "recording_prompts.json"
        old_prompts = json.loads(old_prompts_path.read_text(encoding="utf-8")) if old_prompts_path.is_file() else {}
        for identity, source in (config.recording_prompt_manifest_paths or {}).items():
            if identity != config.rerun_recording_id and identity in old_prompts:
                if json.loads(Path(source).read_text(encoding="utf-8")) != old_prompts[identity]:
                    raise ValueError("unaffected_prompt_changed")
    if config.trial_manifest_path:
        trial_snapshot_source = Path(config.trial_manifest_path)
        for identity, trials in trials_by_record.items():
            exclusions = intervals.loc[intervals.recording_id.astype(str).eq(identity)
                                       & intervals.segment_role.eq("manual_exclusion")]
            prompt_id = (json.loads(Path(config.recording_prompt_manifest_paths[identity]).read_text(
                encoding="utf-8")).get("prompt_id", "")
                if config.recording_prompt_manifest_paths else "")
            for trial in trials:
                if prompt_id and trial["prompt_id"] != prompt_id:
                    raise ValueError("TRIAL_PROMPT_MISMATCH")
                if any(float(row.start_sec) < trial["end_sec"]
                       and float(row.end_sec) > trial["start_sec"]
                       for row in exclusions.itertuples()):
                    raise ValueError("TRIAL_CROSSES_EXCLUSION")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = base / "runs" / run_id
    state_path = run_dir / "logs" / "run_state.json"
    _atomic_json({"alignment_run_id": run_id, "status": "RUNNING",
                  "updated_utc": datetime.now(timezone.utc).isoformat()}, state_path)
    prompt_snapshot = run_dir / "configs" / "prompt_manifest.json"
    _atomic_json(json.loads(Path(config.prompt_manifest_path).read_text(encoding="utf-8")),
                 prompt_snapshot)
    if config.recording_prompt_manifest_paths:
        _atomic_json({identity: json.loads(Path(path).read_text(encoding="utf-8"))
                      for identity, path in config.recording_prompt_manifest_paths.items()},
                     run_dir / "configs" / "recording_prompts.json")
    if config.trial_manifest_path:
        _atomic_json(json.loads(trial_snapshot_source.read_text(encoding="utf-8")),
                     run_dir / "configs" / "alignment_trials.json")
    word_input = _source_rows(config.words_csv, "word") if config.source == "external" else pd.DataFrame()
    phone_input = _source_rows(config.phones_csv, "phone") if config.source == "external" else pd.DataFrame()
    if config.source == "external" and config.words_csv == config.phones_csv:
        phone_input = _source_rows(config.words_csv, "phone")
    word_rows: list[dict] = []
    phone_rows: list[dict] = []
    diagnostics: list[dict] = []
    trial_diagnostics: list[dict] = []
    trial_token_rows: list[dict] = []
    time_rows: list[dict] = []
    provider = config.provider or MFAProvider()
    prepared: dict[str, tuple[list[TimePiece], int]] = {}
    preparation_errors: dict[str, str] = {}
    source_audio_hashes: dict[str, str] = {}
    working_audio_hashes: dict[str, str] = {}
    grids: dict[str, Path] = {}
    unit_to_trial: dict[str, tuple[str, dict]] = {}
    speakers: dict[str, str] = {}
    profile = MfaProfile.load(config.mfa_profile_path) if config.source == "mfa" and config.mfa_profile_path else None
    if profile is not None:
        if config.working_sample_rate_hz != profile.working_sample_rate_hz:
            raise ValueError("MFA_WORKING_AUDIO_PROFILE_MISMATCH")
        _atomic_json(json.loads(Path(config.mfa_profile_path).read_text(encoding="utf-8")),
                     run_dir / "configs" / "mfa_profile.json")
    provider_report: dict = {}
    if config.source == "mfa":
        if progress_callback:
            progress_callback(0, 0, "Preparing corpus...")
        speakers = _speakers(config, kept)
        speaker_sources = {str(item["recording_id"]): str(item.get("speaker_id_source", "unspecified"))
                           for item in json.loads(Path(config.speaker_manifest_path).read_text(
                               encoding="utf-8")).get("recordings", [])}
        transcripts = _transcripts(prompt, config, kept, record_prompts,
                                   run_dir / "configs" / "alignment_transcripts.json")
        corpus = run_dir / "mfa_input"
        for prep_done, record in enumerate(kept.itertuples(), 1):
            identity = str(record.recording_id)
            if config.rerun_recording_id and identity != config.rerun_recording_id:
                continue
            try:
                source = Path(str(record.analysis_wav_path))
                info = sf.info(source)
                start, end = float(record.analysis_start_sec), float(record.analysis_end_sec)
                if not 0 <= start < end <= info.duration + 1e-7:
                    raise ValueError("invalid_reviewed_analysis_bounds")
                record_intervals = intervals.loc[intervals.recording_id.astype(str).eq(identity)]
                for trial in trials_by_record[identity]:
                    spans = _analysis_pieces(record_intervals, trial["start_sec"], trial["end_sec"])
                    # An exclusion is a hard sequence break. Never concatenate
                    # speech on its two sides into one MFA utterance.
                    if (len(spans) != 1 or
                            abs(spans[0][0] - float(trial["start_sec"])) > 1e-7 or
                            abs(spans[0][1] - float(trial["end_sec"])) > 1e-7):
                        raise ValueError(f"TRIAL_CROSSES_EXCLUSION:{trial['trial_id']}")
                    unit = identity if trial["trial_id"] == "legacy_whole_recording" else (
                        f"trial_{len(unit_to_trial) + 1:06d}")
                    speaker_dir = corpus / speakers[identity]
                    working = speaker_dir / f"{unit}.wav"
                    pieces, native_rate = _working_audio(source, spans, config.working_sample_rate_hz,
                                                         working)
                    (speaker_dir / f"{unit}.lab").write_text(transcripts[identity], encoding="utf-8")
                    prepared[unit] = (pieces, native_rate)
                    unit_to_trial[unit] = (identity, trial)
                    working_audio_hashes[unit] = _hash(working)
                    for piece in pieces:
                        time_rows.append({"alignment_run_id": run_id, "recording_id": identity,
                                          "trial_id": trial["trial_id"], **piece.__dict__})
                source_audio_hashes[identity] = _hash(source)
            except (OSError, ValueError) as exc:
                preparation_errors[identity] = str(exc)
            if progress_callback:
                progress_callback(prep_done, len(kept),
                                  f"Recordings prepared: {prep_done} / {len(kept)} · "
                                  f"Trials prepared: {len(prepared)}")
        try:
            if prepared:
                if profile is not None:
                    if progress_callback:
                        progress_callback(0, 0, "Validating transcript and dictionary...")
                    if profile.language != prompt.language or profile.phone_set != prompt.phone_set:
                        raise ValueError("PHONESET_MAPPING_REQUIRED")
                    provider_report = provider.preflight(corpus, profile,
                                                         [transcripts[unit_to_trial[key][0]] for key in prepared],
                                                         run_dir / "logs")
                    _atomic_json(provider_report, run_dir / "logs" / "mfa_preflight.json")
                    if provider_report["status"] != "AVAILABLE":
                        _atomic_json({"alignment_run_id": run_id, "status": "PREFLIGHT_FAILED",
                                      "reason": provider_report["status"],
                                      "updated_utc": datetime.now(timezone.utc).isoformat()}, state_path)
                        raise RuntimeError(provider_report["status"])
                    if progress_callback:
                        progress_callback(0, 0, "Running MFA corpus alignment...")
                    grids = provider.align_corpus(corpus, run_dir / "provider", profile,
                                                  run_dir / "logs")
                    provider_report["diagnostic_paths"] = provider.collect_diagnostics(
                        run_dir / "provider", run_dir / "logs")
                else:
                    # Mock providers exercise the same corpus-level handoff in tests.
                    grids = provider.align_corpus(corpus, run_dir / "provider", None,
                                                  run_dir / "logs")
        finally:
            shutil.rmtree(corpus, ignore_errors=True)
    if progress_callback:
        progress_callback(0, len(kept), "Parsing word and phone boundaries...")
    for done, record in enumerate(kept.itertuples(), 1):
        if config.rerun_recording_id and str(record.recording_id) != config.rerun_recording_id:
            identity = str(record.recording_id)
            for name, output in (("alignment_words", word_rows),
                                 ("alignment_phones", phone_rows),
                                 ("alignment_diagnostics", diagnostics),
                                 ("alignment_trial_diagnostics", trial_diagnostics),
                                 ("alignment_trial_tokens", trial_token_rows)):
                previous = prior_tables[name].loc[
                    prior_tables[name].recording_id.astype(str).eq(identity)].copy()
                if "alignment_run_id" in previous:
                    previous["alignment_run_id"] = run_id
                output.extend(previous.to_dict("records"))
            if identity in prior_manifest.get("source_audio_sha256_by_recording", {}):
                source_audio_hashes[identity] = prior_manifest[
                    "source_audio_sha256_by_recording"][identity]
            if progress_callback:
                progress_callback(done, len(kept), f"Reused reviewed run for {record.file_name}")
            continue
        record_intervals = intervals.loc[intervals.recording_id.astype(str).eq(str(record.recording_id))]
        start, end = float(record.analysis_start_sec), float(record.analysis_end_sec)
        spans = _analysis_pieces(record_intervals, start, end)
        native_rate = 0
        working_rate = config.working_sample_rate_hz if config.source == "mfa" else 0
        rows_w: list[dict] = []
        rows_p: list[dict] = []
        invalid = 0
        outside = 0
        reason = ""
        try:
            record_prompt = record_prompts[str(record.recording_id)]
            audio_path = Path(str(record.analysis_wav_path))
            info = sf.info(audio_path)
            native_rate = info.samplerate
            if not 0 <= start < end <= info.duration + 1e-7:
                raise ValueError("invalid_reviewed_analysis_bounds")
            if not spans:
                raise ValueError("empty_reviewed_analysis_domain")
            if config.source == "mfa" and config.trial_manifest_path:
                identity = str(record.recording_id)
                if identity in preparation_errors:
                    raise ValueError(preparation_errors[identity])
                word_offset = phone_offset = 0
                for trial_number, trial in enumerate(trials_by_record[identity], 1):
                    unit = next((key for key, value in unit_to_trial.items()
                                 if value[0] == identity and value[1]["trial_id"] == trial["trial_id"]), "")
                    if unit not in grids:
                        raise ValueError(f"ALIGNMENT_FAILED:no_textgrid:{trial['trial_id']}")
                    pieces, native_rate = prepared[unit]
                    raw_w, raw_p = (provider.parse_outputs(grids[unit])
                                    if hasattr(provider, "parse_outputs") else _parse_textgrid(grids[unit]))
                    trial_domain = _analysis_pieces(record_intervals, trial["start_sec"],
                                                    trial["end_sec"])
                    trial_w, bad_w, outside_w = _token_rows(
                        raw_w, kind="word", record=record, prompt=record_prompt,
                        run_id=run_id, source=config.source, native_rate=native_rate,
                        pieces=pieces, domain=trial_domain)
                    trial_p, bad_p, outside_p = _token_rows(
                        raw_p, kind="phone", record=record, prompt=record_prompt,
                        run_id=run_id, source=config.source, native_rate=native_rate,
                        pieces=pieces, domain=trial_domain)
                    if bad_w or bad_p or _validate_order(trial_w)[0] or _validate_order(trial_p)[0]:
                        raise ValueError(f"INVALID_BOUNDARIES:{trial['trial_id']}")
                    matched, missing_trial, extra_trial = _coverage(trial_w, record_prompt.expected_words)
                    trial_diagnostics.append({"alignment_run_id": run_id,
                        "recording_id": identity, "trial_id": trial["trial_id"],
                        "prompt_id": trial["prompt_id"], "start_sec": trial["start_sec"],
                        "end_sec": trial["end_sec"], "n_words": len(trial_w),
                        "n_phones": len(trial_p), "expected_word_count": len(record_prompt.expected_words),
                        "aligned_expected_word_count": matched,
                        "word_coverage": matched / len(record_prompt.expected_words),
                        "missing_words": json.dumps(missing_trial), "extra_tokens": extra_trial,
                        "status": "ALIGNED" if trial_w and trial_p and matched / len(record_prompt.expected_words) >= .8
                                  else "REVIEW_REQUIRED"})
                    for word in trial_w:
                        word["word_index"] += word_offset
                        word["sequence_id"] += (trial_number - 1) * 100000
                        trial_token_rows.append({"recording_id": identity,
                            "trial_id": trial["trial_id"], "token_type": "word",
                            "token_index": word["word_index"], "start_sec": word["start_sec"],
                            "end_sec": word["end_sec"]})
                    for phone in trial_p:
                        if phone["word_index"]:
                            phone["word_index"] += word_offset
                        phone["phone_index"] += phone_offset
                        phone["sequence_id"] += (trial_number - 1) * 100000
                        trial_token_rows.append({"recording_id": identity,
                            "trial_id": trial["trial_id"], "token_type": "phone",
                            "token_index": phone["phone_index"], "start_sec": phone["start_sec"],
                            "end_sec": phone["end_sec"]})
                    rows_w.extend(trial_w)
                    rows_p.extend(trial_p)
                    word_offset += len(trial_w)
                    phone_offset += len(trial_p)
                    invalid += bad_w + bad_p
                    outside += outside_w + outside_p
                bad_w = bad_p = outside_w = outside_p = 0
            elif config.source == "mfa":
                identity = str(record.recording_id)
                if identity in preparation_errors:
                    raise ValueError(preparation_errors[identity])
                if identity not in grids:
                    raise ValueError("ALIGNMENT_FAILED:no_textgrid")
                pieces, native_rate = prepared[identity]
                raw_w, raw_p = provider.parse_outputs(grids[identity]) if hasattr(provider, "parse_outputs") else _parse_textgrid(grids[identity])
                for label in raw_p.get("label", pd.Series(dtype=str)):
                    normalize_phone(str(label), record_prompt.phone_set)
            else:
                pieces = None
                raw_w, raw_p = word_input, phone_input
            if not (config.source == "mfa" and config.trial_manifest_path):
                rows_w, bad_w, outside_w = _token_rows(raw_w, kind="word", record=record, prompt=record_prompt,
                                            run_id=run_id, source=config.source, native_rate=native_rate,
                                            pieces=pieces, domain=spans)
                rows_p, bad_p, outside_p = _token_rows(raw_p, kind="phone", record=record, prompt=record_prompt,
                                            run_id=run_id, source=config.source, native_rate=native_rate,
                                            pieces=pieces, domain=spans)
                invalid = bad_w + bad_p
                outside = outside_w + outside_p
            for phone in rows_p:
                if int(phone["word_index"]) == 0:
                    owners = [word for word in rows_w
                              if word["start_sec"] - 1e-7 <= phone["start_sec"]
                              and phone["end_sec"] <= word["end_sec"] + 1e-7]
                    if len(owners) == 1:
                        phone["word_index"] = owners[0]["word_index"]
                    elif config.source == "mfa":
                        invalid += 1
            word_overlap, word_nonmonotonic = _validate_order(rows_w)
            phone_overlap, phone_nonmonotonic = _validate_order(rows_p)
            overlaps = word_overlap + phone_overlap
            nonmonotonic = word_nonmonotonic + phone_nonmonotonic
            rows_w.sort(key=lambda row: row["start_sec"])
            rows_p.sort(key=lambda row: row["start_sec"])
            expected_sequence = record_prompt.expected_words * len(trials_by_record[str(record.recording_id)])
            aligned_expected, missing, extra = _coverage(rows_w, expected_sequence)
            coverage = aligned_expected / len(expected_sequence)
            if invalid or overlaps or nonmonotonic:
                status, reason = "INVALID_BOUNDARIES", "invalid_or_overlapping_tokens"
                rows_w, rows_p = [], []
            elif not rows_w:
                status, reason = "ALIGNMENT_FAILED", "no_aligned_words"
                rows_p = []
            elif coverage < 0.80:
                status, reason = "LOW_COVERAGE", "word_coverage_below_80_percent"
            elif not rows_p:
                status, reason = "PARTIAL_ALIGNMENT", "no_aligned_phones"
            else:
                status = "ALIGNED"
        except Exception as exc:  # per-recording failure still advances progress
            status = "ALIGNMENT_FAILED"
            reason = str(exc)[:500]
            rows_w, rows_p = [], []
            expected_sequence = (record_prompts[str(record.recording_id)].expected_words *
                                 len(trials_by_record[str(record.recording_id)]))
            aligned_expected, missing, extra, coverage = 0, list(expected_sequence), 0, 0.0
            overlaps = nonmonotonic = 0
            outside = 0
        word_rows.extend(rows_w)
        phone_rows.extend(rows_p)
        diagnostics.append({"alignment_run_id": run_id, "recording_id": str(record.recording_id),
                            "file_name": str(record.file_name), "status": status, "reason": reason,
                            "expected_word_count": len(expected_sequence),
                            "aligned_expected_word_count": aligned_expected, "word_coverage": coverage,
                            "missing_words": json.dumps(missing), "extra_tokens": extra,
                            "n_words": len(rows_w), "n_phones": len(rows_p),
                            "invalid_intervals": invalid, "overlapping_intervals": overlaps,
                            "nonmonotonic_intervals": nonmonotonic,
                            "outside_domain_tokens": outside,
                            "canonical_sample_rate_hz": native_rate,
                            "working_sample_rate_hz": working_rate})
        if progress_callback:
            progress_callback(done, len(kept), f"Aligning {done} / {len(kept)} — {record.file_name}")
    tables = run_dir / "tables"
    words_path, phones_path = tables / "alignment_words.csv", tables / "alignment_phones.csv"
    diagnostic_path = tables / "alignment_diagnostics.csv"
    _atomic_csv(pd.DataFrame(word_rows, columns=WORD_COLUMNS), words_path)
    _atomic_csv(pd.DataFrame(phone_rows, columns=PHONE_COLUMNS), phones_path)
    _atomic_csv(pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS), diagnostic_path)
    if config.trial_manifest_path:
        _atomic_csv(pd.DataFrame(trial_diagnostics), tables / "alignment_trial_diagnostics.csv")
        _atomic_csv(pd.DataFrame(trial_token_rows), tables / "alignment_trial_tokens.csv")
        if config.rerun_recording_id and prior_run is not None:
            from .corrections import load_corrections, correction_path
            previous_review = prior_run / "tables" / "alignment_trial_review.csv"
            if previous_review.is_file():
                reviews = pd.read_csv(previous_review, keep_default_na=False)
                reviews = reviews.loc[reviews.recording_id.astype(str).ne(
                    config.rerun_recording_id)].copy()
            else:
                reviews = pd.DataFrame(columns=["recording_id", "trial_id", "review_status", "reviewed_utc"])
            current = pd.DataFrame([{"recording_id": config.rerun_recording_id,
                                     "trial_id": item["trial_id"], "review_status": "UNREVIEWED",
                                     "reviewed_utc": ""} for item in
                                    trials_by_record[config.rerun_recording_id]])
            _atomic_csv(pd.concat([reviews, current], ignore_index=True),
                        tables / "alignment_trial_review.csv")
            corrections = load_corrections(root, config.reuse_run_id)
            corrections = corrections.loc[corrections.recording_id.astype(str).ne(
                config.rerun_recording_id)]
            if not corrections.empty:
                _atomic_csv(corrections, correction_path(root, run_id))
    if time_rows:
        _atomic_csv(pd.DataFrame(time_rows), tables / "working_time_map.csv")
    manifest = {"stage": "alignment", "algorithm_version": ALIGNMENT_VERSION,
                "alignment_run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_segmentation_run_id": sorted(set(kept.segmentation_run_id.astype(str))),
                "source_review_run_id": sorted(set(kept.review_run_id.astype(str))),
                "source_final_decisions_sha256": _hash(decision_path),
                "source_final_intervals_sha256": _hash(interval_path),
                "provider": getattr(provider, "name", config.source) if config.source == "mfa" else "external",
                "provider_version": getattr(provider, "version", ""),
                "provider_mode": getattr(provider, "mode", ""),
                "provider_environment": provider_report,
                "mfa_profile_id": profile.profile_id if profile else "",
                "mfa_profile_sha256": (_hash(Path(config.mfa_profile_path))
                                       if config.mfa_profile_path else ""),
                "speaker_manifest_sha256": (_hash(Path(config.speaker_manifest_path))
                                            if config.speaker_manifest_path else ""),
                "speaker_id_by_recording": speakers,
                "speaker_id_source_by_recording": speaker_sources if config.source == "mfa" else {},
                "speaker_grouping_note": (
                    "MFA speaker grouping can affect normalization and adaptation. "
                    "Technical recording keys do not assert participant identity."
                    if config.source == "mfa" else ""),
                "trial_corpus_unit_map": {key: {"recording_id": value[0],
                    "trial_id": value[1]["trial_id"]} for key, value in unit_to_trial.items()},
                "trial_manifest_sha256": (_hash(run_dir / "configs" / "alignment_trials.json")
                                          if config.trial_manifest_path else ""),
                "trial_diagnostics_sha256": (_hash(tables / "alignment_trial_diagnostics.csv")
                                             if config.trial_manifest_path else ""),
                "trial_tokens_sha256": (_hash(tables / "alignment_trial_tokens.csv")
                                        if config.trial_manifest_path else ""),
                "trial_contract_version": "reviewed_trials_v1" if config.trial_manifest_path else "legacy_single_unit",
                "reused_alignment_run_id": config.reuse_run_id,
                "rerun_recording_id": config.rerun_recording_id,
                "reused_prior_manifest_sha256": (_hash(prior_run / "logs" / "stage_manifest.json")
                                                 if prior_run else ""),
                "observed_trials_by_recording": {identity: len(items)
                                                 for identity, items in trials_by_record.items()},
                "expected_trials_by_recording": {
                    identity: json.loads(Path(path).read_text(encoding="utf-8")).get(
                        "expected_repetitions")
                    for identity, path in (config.recording_prompt_manifest_paths or {
                        identity: config.prompt_manifest_path for identity in trials_by_record
                    }).items()},
                "protocol_repetition_shortfall_by_recording": {
                    identity: max(0, int(expected) - len(trials_by_record[identity]))
                    for identity, path in (config.recording_prompt_manifest_paths or {
                        identity: config.prompt_manifest_path for identity in trials_by_record
                    }).items()
                    if (expected := json.loads(Path(path).read_text(
                        encoding="utf-8")).get("expected_repetitions")) is not None},
                "alignment_transcripts_sha256": (_hash(run_dir / "configs" / "alignment_transcripts.json")
                                                 if config.source == "mfa" else ""),
                "normalization_profile_id": "vslp_prompt_nfkc_v1" if config.source == "mfa" else "",
                "provider_commands": getattr(provider, "commands", []),
                "source_audio_sha256_by_recording": source_audio_hashes,
                "working_audio_sha256_by_recording": working_audio_hashes,
                "provider_diagnostic_sha256": {str(path): _hash(Path(path)) for path in
                                               provider_report.get("diagnostic_paths", [])},
                "mfa_textgrid_sha256_by_recording": {identity: _hash(path) for identity, path in grids.items()},
                "acoustic_model": profile.acoustic_model if profile else config.acoustic_model,
                "acoustic_model_version": (profile.acoustic_model_version if profile
                                           else config.acoustic_model_version),
                "dictionary": profile.dictionary if profile else config.dictionary,
                "dictionary_version": (profile.dictionary_version if profile
                                       else config.dictionary_version),
                "task_id": prompt.task_id, "language": prompt.language, "phone_set": prompt.phone_set,
                "prompt_manifest_sha256": _hash(prompt_snapshot),
                "prompt_manifest_snapshot": str(prompt_snapshot),
                "prompt_version": prompt.prompt_version,
                "prompt_version_by_recording": {identity: item.prompt_version
                                                for identity, item in record_prompts.items()},
                "prompt_id_by_recording": {
                    identity: json.loads(Path(path).read_text(encoding="utf-8")).get("prompt_id", "")
                    for identity, path in (config.recording_prompt_manifest_paths or {}).items()},
                "expected_words_by_recording": {identity: list(item.expected_words * len(trials_by_record[identity]))
                                                 for identity, item in record_prompts.items()},
                "source_words_sha256": (_hash(Path(config.words_csv)) if config.words_csv else ""),
                "source_phones_sha256": (_hash(Path(config.phones_csv)) if config.phones_csv else ""),
                "expected_words": list(prompt.expected_words),
                "working_sample_rate_hz": working_rate,
                "working_audio_profile_id": ("mono_mean_polyphase_concat_exclusions_v1"
                                             if config.source == "mfa" else ""),
                "canonical_sample_rate_policy": "native_rate_preserved",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "mfa_validation_log_sha256": (_hash(run_dir / "logs" / "mfa_validation.log")
                                               if (run_dir / "logs" / "mfa_validation.log").exists() else ""),
                "mfa_alignment_log_sha256": (_hash(run_dir / "logs" / "mfa_alignment.log")
                                              if (run_dir / "logs" / "mfa_alignment.log").exists() else ""),
                "number_attempted": len(kept),
                "number_aligned": sum(row["status"] == "ALIGNED" for row in diagnostics),
                "number_partial": sum(row["status"] == "PARTIAL_ALIGNMENT" for row in diagnostics),
                "number_failed": sum(row["status"] in {"ALIGNMENT_FAILED", "INVALID_BOUNDARIES"}
                                     for row in diagnostics),
                "number_below_coverage_threshold": sum(row["status"] == "LOW_COVERAGE"
                                                       for row in diagnostics),
                "words_sha256": _hash(words_path), "phones_sha256": _hash(phones_path),
                "words_path": str(words_path), "phones_path": str(phones_path),
                "diagnostics_path": str(diagnostic_path)}
    if config.trial_manifest_path:
        from .review_policy import ALIGNMENT_REVIEW_POLICY_VERSION, trial_exception_flags

        review_path = tables / "alignment_trial_review.csv"
        previous = (pd.read_csv(review_path, keep_default_na=False)
                    if review_path.is_file() else pd.DataFrame())
        words_frame = pd.DataFrame(word_rows, columns=WORD_COLUMNS)
        phones_frame = pd.DataFrame(phone_rows, columns=PHONE_COLUMNS)
        links_frame = pd.DataFrame(trial_token_rows, columns=[
            "recording_id", "trial_id", "token_type", "token_index",
            "start_sec", "end_sec"])
        review_rows = []
        for diagnostic in trial_diagnostics:
            identity, trial_id = str(diagnostic["recording_id"]), str(diagnostic["trial_id"])
            if config.rerun_recording_id and identity != config.rerun_recording_id:
                continue
            trial = next(item for item in trials_by_record[identity]
                         if item["trial_id"] == trial_id)
            links_for_trial = links_frame.loc[
                links_frame.recording_id.astype(str).eq(identity)
                & links_frame.trial_id.astype(str).eq(trial_id)]
            transcript_overridden = (config.source == "mfa" and
                transcripts[identity] != normalize_transcript(record_prompts[identity].transcript))
            expected = manifest["expected_trials_by_recording"].get(identity)
            flags = trial_exception_flags(
                diagnostic, trial, expected_count=int(expected) if expected is not None else None,
                observed_count=len(trials_by_record[identity]),
                transcript_overridden=transcript_overridden, links=links_for_trial,
                words=words_frame, phones=phones_frame,
                working_sample_rate_hz=int(manifest["working_sample_rate_hz"]))
            auto = config.source == "mfa" and config.structural_auto_review and not flags
            review_rows.append({"recording_id": identity, "trial_id": trial_id,
                                "review_status": ("AUTO_ACCEPTED_STRUCTURAL" if auto else
                                                  "NEEDS_REVIEW" if flags else "UNREVIEWED"),
                                "reviewed_utc": datetime.now(timezone.utc).isoformat() if auto else "",
                                "review_mode": "AUTO" if auto else "",
                                "queued_for_review": not auto,
                                "review_policy_version": ALIGNMENT_REVIEW_POLICY_VERSION,
                                "flags_triggered": json.dumps(flags)})
        if not previous.empty:
            replaced = {(row["recording_id"], row["trial_id"]) for row in review_rows}
            previous = previous.loc[~previous.apply(
                lambda row: (str(row.recording_id), str(row.trial_id)) in replaced,
                axis=1)]
        _atomic_csv(pd.concat([previous, pd.DataFrame(review_rows)], ignore_index=True),
                    review_path)
        manifest["alignment_review_policy_version"] = ALIGNMENT_REVIEW_POLICY_VERSION
    manifest_path = run_dir / "logs" / "stage_manifest.json"
    if progress_callback and config.source == "mfa":
        progress_callback(0, 0, "Validating Alignment...")
    if config.trial_manifest_path:
        from .qc import create_alignment_qc
        qc_paths = create_alignment_qc(root, run_dir, manifest)
        manifest["alignment_qc_artifacts"] = {key: {"path": path, "sha256": _hash(Path(path))}
                                               for key, path in qc_paths.items()}
    _atomic_json(manifest, manifest_path)
    registry_path = base / "logs" / "alignment_registry.json"
    registry = {"runs": list_alignment_runs(root)}
    registry["runs"].append({"alignment_run_id": run_id,
                             "created_utc": manifest["created_utc"],
                             "source": config.source, "manifest_path": str(manifest_path)})
    _atomic_json(registry, registry_path)
    status = "completed" if manifest["number_aligned"] == len(kept) else "completed_with_warnings"
    _atomic_json({"alignment_run_id": run_id,
                  "status": "COMPLETED" if status == "completed" else "COMPLETED_WITH_FLAGS",
                  "updated_utc": datetime.now(timezone.utc).isoformat()}, state_path)
    return StageResult(status, manifest_path, diagnostic_path)


def freeze_alignment(output_root: str | Path, alignment_run_id: str,
                     supersede: bool = False) -> StageResult:
    root = Path(output_root)
    base, final = _paths(root)
    run_dir = base / "runs" / alignment_run_id
    manifest_path = run_dir / "logs" / "stage_manifest.json"
    replacing = (final / "final_alignment_manifest.json").exists()
    if replacing and not supersede:
        raise FileExistsError("frozen_alignment_is_immutable")
    if replacing:
        old_store, issue = load_final_alignment(root, require_current_trials=False)
        if old_store is None:
            raise ValueError(f"existing_frozen_alignment_invalid:{issue}")
    if not manifest_path.is_file():
        raise FileNotFoundError("alignment_run_not_found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = root / "acoustic" / "003_segmentation_review" / "final"
    if (manifest["source_final_decisions_sha256"] != _hash(review / "final_segmentation_decisions.csv")
            or manifest["source_final_intervals_sha256"] != _hash(review / "final_segmentation_intervals.csv")):
        raise ValueError("reviewed_segmentation_changed_since_alignment")
    if manifest["number_aligned"] + manifest.get("number_partial", 0) == 0:
        raise ValueError("no_successful_alignment_to_freeze")
    if manifest.get("trial_contract_version") == "reviewed_trials_v1":
        if manifest.get("trial_manifest_sha256") != _hash(
                run_dir / "configs" / "alignment_trials.json"):
            raise ValueError("trial_plan_hash_mismatch")
        trial_diagnostics_path = run_dir / "tables" / "alignment_trial_diagnostics.csv"
        trial_tokens_path = run_dir / "tables" / "alignment_trial_tokens.csv"
        review_path = run_dir / "tables" / "alignment_trial_review.csv"
        if (manifest.get("trial_diagnostics_sha256") != _hash(trial_diagnostics_path)
                or manifest.get("trial_tokens_sha256") != _hash(trial_tokens_path)):
            raise ValueError("trial_alignment_hash_mismatch")
        if not review_path.is_file():
            raise ValueError("alignment_trial_review_required")
        trial_diagnostics = pd.read_csv(trial_diagnostics_path, keep_default_na=False)
        trial_plan = json.loads((run_dir / "configs" / "alignment_trials.json").read_text(
            encoding="utf-8"))["trials"]
        if len(trial_diagnostics) != len(trial_plan):
            raise ValueError("alignment_trial_results_incomplete")
        reviews = pd.read_csv(review_path, keep_default_na=False)
        expected_keys = set(zip(trial_diagnostics.recording_id.astype(str),
                                trial_diagnostics.trial_id.astype(str)))
        actual_keys = list(zip(reviews.recording_id.astype(str), reviews.trial_id.astype(str)))
        if (set(actual_keys) != expected_keys or len(actual_keys) != len(set(actual_keys))
                or not reviews.review_status.isin(["ACCEPTED", "AUTO_ACCEPTED_STRUCTURAL"]).all()):
            raise ValueError("alignment_trial_review_required:all_alignment_trials_must_be_accepted")
    words = pd.read_csv(manifest["words_path"], keep_default_na=False)
    phones = pd.read_csv(manifest["phones_path"], keep_default_na=False)
    diagnostics = pd.read_csv(manifest["diagnostics_path"], keep_default_na=False)
    if not set(WORD_COLUMNS) <= set(words) or not set(PHONE_COLUMNS) <= set(phones):
        raise ValueError("alignment_schema_missing")
    if manifest["words_sha256"] != _hash(Path(manifest["words_path"])) or manifest["phones_sha256"] != _hash(Path(manifest["phones_path"])):
        raise ValueError("alignment_run_hash_mismatch")
    from .corrections import load_corrections, reviewed_tables
    corrections = load_corrections(root, alignment_run_id)
    if not corrections.empty:
        if not corrections.reviewer_state.eq("APPROVED").all():
            raise ValueError("manual_alignment_corrections_require_acceptance")
        words, phones = reviewed_tables(root, alignment_run_id)
    _validate_frozen_structure(words, phones, diagnostics, root, alignment_run_id, manifest)
    if manifest.get("trial_contract_version") == "reviewed_trials_v1":
        links = pd.read_csv(trial_tokens_path, keep_default_na=False)
        token_rows = {}
        for kind, source, index_name in (("word", words, "word_index"),
                                         ("phone", phones, "phone_index")):
            linked = links.loc[links.token_type.eq(kind)]
            expected = set(zip(source.recording_id.astype(str), source[index_name].astype(int)))
            actual = list(zip(linked.recording_id.astype(str), linked.token_index.astype(int)))
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError("trial_token_identity_mismatch")
            token_rows.update({(kind, str(row.recording_id), int(getattr(row, index_name))): row
                               for row in source.itertuples()})
        bounds = {(str(item["recording_id"]), str(item["trial_id"])): item
                  for item in trial_plan}
        # Original-time remapping from 16-kHz working samples can differ from a
        # confirmed trial edge by less than half one working sample. No clipping.
        tolerance = 0.5 / int(manifest["working_sample_rate_hz"])
        for item in links.itertuples():
            trial = bounds.get((str(item.recording_id), str(item.trial_id)))
            token = token_rows[(str(item.token_type), str(item.recording_id),
                                int(item.token_index))]
            start, end = float(token.start_sec), float(token.end_sec)
            if not trial or not (float(trial["start_sec"]) - tolerance <= start
                                  < end <= float(trial["end_sec"]) + tolerance):
                detail = {"recording_id": str(item.recording_id),
                          "file_name": str(token.file_name), "trial_id": str(item.trial_id),
                          "tier": str(item.token_type), "token_index": int(item.token_index),
                          "trial_start_sec": None if not trial else float(trial["start_sec"]),
                          "trial_end_sec": None if not trial else float(trial["end_sec"]),
                          "token_start_sec": start, "token_end_sec": end,
                          "source": str(token.alignment_source)}
                raise ValueError("trial_token_boundary_mismatch:" + json.dumps(detail))
    destination = base / f".final_pending_{uuid4().hex}" if replacing else final
    words_out = destination / "final_alignment_words.csv"
    phones_out = destination / "final_alignment_phones.csv"
    diagnostics_out = destination / "final_alignment_diagnostics.csv"
    _atomic_csv(words, words_out)
    _atomic_csv(phones, phones_out)
    _atomic_csv(diagnostics, diagnostics_out)
    if manifest.get("trial_contract_version") == "reviewed_trials_v1":
        _atomic_csv(pd.DataFrame(trial_plan), destination / "final_alignment_trials.csv")
        for source_name, final_name in (
                ("alignment_trial_diagnostics.csv", "final_alignment_trial_diagnostics.csv"),
                ("alignment_trial_tokens.csv", "final_alignment_trial_tokens.csv"),
                ("alignment_trial_review.csv", "final_alignment_trial_review.csv")):
            _atomic_csv(pd.read_csv(run_dir / "tables" / source_name, keep_default_na=False),
                        destination / final_name)
        final_links = pd.read_csv(destination / "final_alignment_trial_tokens.csv",
                                  keep_default_na=False)
        for kind, source in (("word", words), ("phone", phones)):
            subset = source.set_index(["recording_id", f"{kind}_index"])
            for index, link in final_links.loc[final_links.token_type.eq(kind)].iterrows():
                key = (str(link.recording_id), int(link.token_index))
                if key in subset.index:
                    final_links.at[index, "start_sec"] = subset.loc[key, "start_sec"]
                    final_links.at[index, "end_sec"] = subset.loc[key, "end_sec"]
        _atomic_csv(final_links, destination / "final_alignment_trial_tokens.csv")
        final_reviews = pd.read_csv(destination / "final_alignment_trial_review.csv",
                                    keep_default_na=False)
        manual_keys = set(zip(corrections.recording_id.astype(str),
                              corrections.trial_id.astype(str)))
        final_reviews["final_status"] = [
            "ACCEPTED_MANUAL" if (str(row.recording_id), str(row.trial_id)) in manual_keys
            else "AUTO_ACCEPTED_STRUCTURAL" if row.review_status == "AUTO_ACCEPTED_STRUCTURAL"
            else "ACCEPTED_MFA" for row in final_reviews.itertuples()]
        _atomic_csv(final_reviews, destination / "final_alignment_trial_review.csv")
        _atomic_csv(corrections, destination / "final_alignment_manual_corrections.csv")
        from .qc import create_alignment_qc
        final_qc = create_alignment_qc(root, run_dir, manifest, final_dir=destination)
    result = {**manifest, "alignment_status": "FROZEN",
              "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "final_words_sha256": _hash(words_out), "final_phones_sha256": _hash(phones_out),
              "final_diagnostics_sha256": _hash(diagnostics_out),
              "final_words_path": str(final / words_out.name),
              "final_phones_path": str(final / phones_out.name),
              "final_diagnostics_path": str(final / diagnostics_out.name),
              "supersedes_alignment_run_id": (old_store.manifest["alignment_run_id"]
                                               if replacing else "")}
    if manifest.get("trial_contract_version") == "reviewed_trials_v1":
        result["final_trial_artifact_sha256"] = {
            name: _hash(destination / name) for name in (
                "final_alignment_trials.csv",
                "final_alignment_trial_diagnostics.csv", "final_alignment_trial_tokens.csv",
                "final_alignment_trial_review.csv",
                "final_alignment_manual_corrections.csv")}
        result["final_trial_tokens_path"] = str(final / "final_alignment_trial_tokens.csv")
        result["final_trial_diagnostics_path"] = str(final / "final_alignment_trial_diagnostics.csv")
        result["final_trial_review_path"] = str(final / "final_alignment_trial_review.csv")
        result["final_manual_corrections_path"] = str(
            final / "final_alignment_manual_corrections.csv")
        result["reviewed_boundary_source"] = "mfa_plus_approved_manual_corrections_v1"
        result["final_alignment_qc_artifacts"] = {
            key: {"path": str(final / Path(path).relative_to(destination)),
                  "sha256": _hash(Path(path))}
            for key, path in final_qc.items()}
    final_manifest = destination / "final_alignment_manifest.json"
    _atomic_json(result, final_manifest)
    if replacing:
        archive = base / "archived_final" / (
            f"{old_store.manifest['alignment_run_id']}_{uuid4().hex[:8]}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        final.rename(archive)
        try:
            destination.rename(final)
        except OSError:
            archive.rename(final)
            raise
        # Archived data are retained; update path references for local reload.
        archived_manifest_path = archive / "final_alignment_manifest.json"
        archived_manifest = json.loads(archived_manifest_path.read_text(encoding="utf-8"))
        for key in ("final_words_path", "final_phones_path", "final_diagnostics_path",
                    "final_trial_tokens_path", "final_trial_diagnostics_path",
                    "final_trial_review_path", "final_manual_corrections_path"):
            if archived_manifest.get(key):
                archived_manifest[key] = str(archive / Path(archived_manifest[key]).name)
        for artifact in archived_manifest.get("final_alignment_qc_artifacts", {}).values():
            if artifact.get("path"):
                artifact["path"] = str(archive / Path(artifact["path"]).relative_to(final))
        _atomic_json(archived_manifest, archived_manifest_path)
    return StageResult("completed", final / "final_alignment_manifest.json",
                       final / "final_alignment_diagnostics.csv")


def _validate_frozen_structure(words: pd.DataFrame, phones: pd.DataFrame,
                               diagnostics: pd.DataFrame, root: Path,
                               run_id: str, manifest: dict) -> None:
    decisions = pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" /
                            "final_segmentation_decisions.csv", keep_default_na=False)
    intervals = pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" /
                            "final_segmentation_intervals.csv", keep_default_na=False)
    by_record = {str(row.recording_id): row for row in decisions.itertuples()}
    if diagnostics.recording_id.astype(str).duplicated().any():
        raise ValueError("duplicate_alignment_diagnostic")
    for table in (words, phones):
        for identity, group in table.groupby(table.recording_id.astype(str)):
            if identity not in by_record:
                raise ValueError("unknown_alignment_recording")
            record = by_record[identity]
            if (not group.alignment_run_id.astype(str).eq(run_id).all()
                    or not group.task_id.astype(str).eq(str(manifest["task_id"])).all()
                    or not group.prompt_version.astype(str).eq(str(
                        manifest.get("prompt_version_by_recording", {}).get(
                            identity, manifest["prompt_version"]))).all()):
                raise ValueError("alignment_identity_mismatch")
            domain = _analysis_pieces(intervals.loc[intervals.recording_id.astype(str).eq(identity)],
                                      float(record.analysis_start_sec), float(record.analysis_end_sec))
            rows = group.sort_values("start_sec").to_dict("records")
            if _validate_order(rows)[0]:
                raise ValueError("alignment_overlap")
            for row in rows:
                start, end = float(row["start_sec"]), float(row["end_sec"])
                if (not np.isfinite([start, end]).all() or start < 0 or end <= start
                        or not any(left - 1e-6 <= start < end <= right + 1e-6
                                   for left, right in domain)):
                    raise ValueError("alignment_boundary_outside_reviewed_domain")
    for phone in phones.itertuples():
        owners = words.loc[words.recording_id.astype(str).eq(str(phone.recording_id))
                           & words.word_index.astype(int).eq(int(phone.word_index))]
        if len(owners) != 1 or not (float(owners.iloc[0].start_sec) - 1e-6 <= float(phone.start_sec)
                                    < float(phone.end_sec) <= float(owners.iloc[0].end_sec) + 1e-6):
            raise ValueError("phone_word_relationship_invalid")


@dataclass(frozen=True)
class AlignmentStore:
    words: pd.DataFrame
    phones: pd.DataFrame
    diagnostics: pd.DataFrame
    manifest: dict

    def get_word_tokens(self, recording_id: str) -> pd.DataFrame:
        return self.words.loc[self.words.recording_id.astype(str).eq(str(recording_id))].copy()

    def get_phone_tokens(self, recording_id: str) -> pd.DataFrame:
        return self.phones.loc[self.phones.recording_id.astype(str).eq(str(recording_id))].copy()

    def get_vowel_tokens(self, recording_id: str) -> pd.DataFrame:
        phones = self.get_phone_tokens(recording_id)
        return phones.loc[phones.is_vowel.astype(str).str.lower().isin(["true", "1"])].copy()

    def get_tokens_by_label(self, recording_id: str, label: str) -> pd.DataFrame:
        phones = self.get_phone_tokens(recording_id)
        return phones.loc[phones.phone_normalized.astype(str).eq(label.upper())].copy()

    def get_trial_tokens(self, recording_id: str, trial_id: str,
                         token_type: str = "word") -> pd.DataFrame:
        """Resolve a frozen trial through its auxiliary identity map."""
        path = self.manifest.get("final_trial_tokens_path", "")
        if not path:
            return pd.DataFrame()
        mapping = pd.read_csv(path, keep_default_na=False)
        indices = mapping.loc[mapping.recording_id.astype(str).eq(str(recording_id))
                              & mapping.trial_id.astype(str).eq(str(trial_id))
                              & mapping.token_type.eq(token_type), "token_index"]
        source = self.words if token_type == "word" else self.phones
        index_name = "word_index" if token_type == "word" else "phone_index"
        return source.loc[source.recording_id.astype(str).eq(str(recording_id))
                          & source[index_name].isin(indices)].copy()

    def family07_rows(self) -> pd.DataFrame:
        words = self.words.rename(columns={"word": "label"}).copy()
        words["token_type"] = "word"
        vowels = self.phones.loc[
            self.phones.is_vowel.astype(str).str.lower().isin(["true", "1"])
        ].rename(columns={"phone": "label"}).copy()
        vowels["token_type"] = "vowel"
        combined = pd.concat([words, vowels], ignore_index=True)
        expected = self.diagnostics.set_index("recording_id").expected_word_count.to_dict()
        combined["expected_count"] = combined.recording_id.map(expected)
        combined["valid"] = True
        return combined


def load_final_alignment(output_root: str | Path, *,
                         require_current_trials: bool = True) -> tuple[AlignmentStore | None, str]:
    root = Path(output_root)
    _, final = _paths(root)
    manifest_path = final / "final_alignment_manifest.json"
    if not manifest_path.is_file():
        return None, "missing_alignment"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        review = root / "acoustic" / "003_segmentation_review" / "final"
        if (manifest["source_final_decisions_sha256"] != _hash(review / "final_segmentation_decisions.csv")
                or manifest["source_final_intervals_sha256"] != _hash(review / "final_segmentation_intervals.csv")):
            return None, "alignment_review_source_mismatch"
        choices_path = root / "configs" / "alignment_choices.json"
        if (require_current_trials and manifest.get("trial_contract_version") == "reviewed_trials_v1"
                and choices_path.is_file()):
            choices = json.loads(choices_path.read_text(encoding="utf-8"))
            if choices.get("pending_trial_revision"):
                return None, "stale_alignment_trial_revision"
            plan_path = (root / "acoustic" / "004_alignment" / "runs" /
                         manifest["alignment_run_id"] / "configs" / "alignment_trials.json")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            frozen_trials = {(str(item["recording_id"]), str(item["trial_id"])): item
                             for item in plan["trials"]}
            current_trials = {(str(identity), str(item["trial_id"])): item
                              for identity, items in choices.get("trials", {}).items()
                              for item in items}
            if current_trials and (set(current_trials) != set(frozen_trials) or any(
                    current_trials[key].get("prompt_id", "") != frozen_trials[key].get("prompt_id", "")
                    or abs(float(current_trials[key]["start_sec"]) -
                           float(frozen_trials[key]["start_sec"])) > 1e-9
                    or abs(float(current_trials[key]["end_sec"]) -
                           float(frozen_trials[key]["end_sec"])) > 1e-9
                    for key in current_trials)):
                return None, "stale_alignment_trial_revision"
        words_path = final / "final_alignment_words.csv"
        phones_path = final / "final_alignment_phones.csv"
        diagnostics_path = final / "final_alignment_diagnostics.csv"
        if (manifest["final_words_sha256"] != _hash(words_path)
                or manifest["final_phones_sha256"] != _hash(phones_path)
                or manifest["final_diagnostics_sha256"] != _hash(diagnostics_path)):
            return None, "alignment_hash_mismatch"
        for name, expected in manifest.get("final_trial_artifact_sha256", {}).items():
            if _hash(final / name) != expected:
                return None, "alignment_trial_hash_mismatch"
        for artifact in manifest.get("final_alignment_qc_artifacts", {}).values():
            if _hash(Path(artifact["path"])) != artifact["sha256"]:
                return None, "alignment_qc_hash_mismatch"
        words = pd.read_csv(words_path, keep_default_na=False)
        phones = pd.read_csv(phones_path, keep_default_na=False)
        diagnostics = pd.read_csv(diagnostics_path, keep_default_na=False)
        if not set(WORD_COLUMNS) <= set(words) or not set(PHONE_COLUMNS) <= set(phones):
            return None, "alignment_schema_missing"
        return AlignmentStore(words, phones, diagnostics, manifest), ""
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None, "invalid_final_alignment"
