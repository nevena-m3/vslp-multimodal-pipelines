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
    record_prompts = {str(identity): prompt for identity in kept.recording_id.astype(str)}
    if config.recording_prompt_manifest_paths:
        for identity, source in config.recording_prompt_manifest_paths.items():
            if identity not in record_prompts:
                raise ValueError("unknown_recording_prompt_mapping")
            individual = PromptManifest.load(source)
            if individual.task_id != prompt.task_id or individual.language != prompt.language:
                raise ValueError("PROMPT_MISMATCH")
            record_prompts[identity] = individual
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
    word_input = _source_rows(config.words_csv, "word") if config.source == "external" else pd.DataFrame()
    phone_input = _source_rows(config.phones_csv, "phone") if config.source == "external" else pd.DataFrame()
    if config.source == "external" and config.words_csv == config.phones_csv:
        phone_input = _source_rows(config.words_csv, "phone")
    word_rows: list[dict] = []
    phone_rows: list[dict] = []
    diagnostics: list[dict] = []
    time_rows: list[dict] = []
    provider = config.provider or MFAProvider()
    prepared: dict[str, tuple[list[TimePiece], int]] = {}
    preparation_errors: dict[str, str] = {}
    source_audio_hashes: dict[str, str] = {}
    working_audio_hashes: dict[str, str] = {}
    grids: dict[str, Path] = {}
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
        transcripts = _transcripts(prompt, config, kept, record_prompts,
                                   run_dir / "configs" / "alignment_transcripts.json")
        corpus = run_dir / "mfa_input"
        for record in kept.itertuples():
            identity = str(record.recording_id)
            try:
                source = Path(str(record.analysis_wav_path))
                info = sf.info(source)
                start, end = float(record.analysis_start_sec), float(record.analysis_end_sec)
                if not 0 <= start < end <= info.duration + 1e-7:
                    raise ValueError("invalid_reviewed_analysis_bounds")
                record_intervals = intervals.loc[intervals.recording_id.astype(str).eq(identity)]
                spans = _analysis_pieces(record_intervals, start, end)
                if not spans:
                    raise ValueError("empty_reviewed_analysis_domain")
                speaker_dir = corpus / speakers[identity]
                working = speaker_dir / f"{identity}.wav"
                pieces, native_rate = _working_audio(source, spans, config.working_sample_rate_hz,
                                                     working)
                (speaker_dir / f"{identity}.lab").write_text(transcripts[identity], encoding="utf-8")
                prepared[identity] = (pieces, native_rate)
                source_audio_hashes[identity] = _hash(source)
                working_audio_hashes[identity] = _hash(working)
                for piece in pieces:
                    time_rows.append({"alignment_run_id": run_id, "recording_id": identity,
                                      **piece.__dict__})
            except (OSError, ValueError) as exc:
                preparation_errors[identity] = str(exc)
        try:
            if prepared:
                if profile is not None:
                    if progress_callback:
                        progress_callback(0, 0, "Validating transcript and dictionary...")
                    if profile.language != prompt.language or profile.phone_set != prompt.phone_set:
                        raise ValueError("PHONESET_MAPPING_REQUIRED")
                    provider_report = provider.preflight(corpus, profile,
                                                         [transcripts[key] for key in prepared],
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
        record_intervals = intervals.loc[intervals.recording_id.astype(str).eq(str(record.recording_id))]
        start, end = float(record.analysis_start_sec), float(record.analysis_end_sec)
        spans = _analysis_pieces(record_intervals, start, end)
        native_rate = 0
        working_rate = config.working_sample_rate_hz if config.source == "mfa" else 0
        rows_w: list[dict] = []
        rows_p: list[dict] = []
        invalid = 0
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
            if config.source == "mfa":
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
            aligned_expected, missing, extra = _coverage(rows_w, record_prompt.expected_words)
            coverage = aligned_expected / len(record_prompt.expected_words)
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
            aligned_expected, missing, extra, coverage = 0, list(record_prompts[str(record.recording_id)].expected_words), 0, 0.0
            overlaps = nonmonotonic = 0
            outside = 0
        word_rows.extend(rows_w)
        phone_rows.extend(rows_p)
        diagnostics.append({"alignment_run_id": run_id, "recording_id": str(record.recording_id),
                            "file_name": str(record.file_name), "status": status, "reason": reason,
                            "expected_word_count": len(record_prompts[str(record.recording_id)].expected_words),
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
                "expected_words_by_recording": {identity: list(item.expected_words)
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
    manifest_path = run_dir / "logs" / "stage_manifest.json"
    if progress_callback and config.source == "mfa":
        progress_callback(0, 0, "Validating Alignment...")
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


def freeze_alignment(output_root: str | Path, alignment_run_id: str) -> StageResult:
    root = Path(output_root)
    base, final = _paths(root)
    run_dir = base / "runs" / alignment_run_id
    manifest_path = run_dir / "logs" / "stage_manifest.json"
    if final.exists() and (final / "final_alignment_manifest.json").exists():
        raise FileExistsError("frozen_alignment_is_immutable")
    if not manifest_path.is_file():
        raise FileNotFoundError("alignment_run_not_found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = root / "acoustic" / "003_segmentation_review" / "final"
    if (manifest["source_final_decisions_sha256"] != _hash(review / "final_segmentation_decisions.csv")
            or manifest["source_final_intervals_sha256"] != _hash(review / "final_segmentation_intervals.csv")):
        raise ValueError("reviewed_segmentation_changed_since_alignment")
    if manifest["number_aligned"] + manifest.get("number_partial", 0) == 0:
        raise ValueError("no_successful_alignment_to_freeze")
    words = pd.read_csv(manifest["words_path"], keep_default_na=False)
    phones = pd.read_csv(manifest["phones_path"], keep_default_na=False)
    diagnostics = pd.read_csv(manifest["diagnostics_path"], keep_default_na=False)
    if not set(WORD_COLUMNS) <= set(words) or not set(PHONE_COLUMNS) <= set(phones):
        raise ValueError("alignment_schema_missing")
    if manifest["words_sha256"] != _hash(Path(manifest["words_path"])) or manifest["phones_sha256"] != _hash(Path(manifest["phones_path"])):
        raise ValueError("alignment_run_hash_mismatch")
    _validate_frozen_structure(words, phones, diagnostics, root, alignment_run_id, manifest)
    words_out = final / "final_alignment_words.csv"
    phones_out = final / "final_alignment_phones.csv"
    diagnostics_out = final / "final_alignment_diagnostics.csv"
    _atomic_csv(words, words_out)
    _atomic_csv(phones, phones_out)
    _atomic_csv(diagnostics, diagnostics_out)
    result = {**manifest, "alignment_status": "FROZEN",
              "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "final_words_sha256": _hash(words_out), "final_phones_sha256": _hash(phones_out),
              "final_diagnostics_sha256": _hash(diagnostics_out),
              "final_words_path": str(words_out), "final_phones_path": str(phones_out),
              "final_diagnostics_path": str(diagnostics_out)}
    final_manifest = final / "final_alignment_manifest.json"
    _atomic_json(result, final_manifest)
    return StageResult("completed", final_manifest, diagnostics_out)


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


def load_final_alignment(output_root: str | Path) -> tuple[AlignmentStore | None, str]:
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
        words_path = final / "final_alignment_words.csv"
        phones_path = final / "final_alignment_phones.csv"
        diagnostics_path = final / "final_alignment_diagnostics.csv"
        if (manifest["final_words_sha256"] != _hash(words_path)
                or manifest["final_phones_sha256"] != _hash(phones_path)
                or manifest["final_diagnostics_sha256"] != _hash(diagnostics_path)):
            return None, "alignment_hash_mismatch"
        words = pd.read_csv(words_path, keep_default_na=False)
        phones = pd.read_csv(phones_path, keep_default_na=False)
        diagnostics = pd.read_csv(diagnostics_path, keep_default_na=False)
        if not set(WORD_COLUMNS) <= set(words) or not set(PHONE_COLUMNS) <= set(phones):
            return None, "alignment_schema_missing"
        return AlignmentStore(words, phones, diagnostics, manifest), ""
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None, "invalid_final_alignment"
