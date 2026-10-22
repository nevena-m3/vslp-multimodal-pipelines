"""Persistent, task-first MFA jobs independent of Acoustic project ingestion.

The workbench reads only audio headers at ingest. It never treats a discovered
recording as reviewed or finalized alignment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from uuid import uuid4

import soundfile as sf

from .cancellable_runner import AlignmentCancelled
from .mfa_provider import MfaProfile, MfaProvider, parse_textgrid, validate_corpus_pairs
from .profiles import default_profile_path
from .stage import normalize_transcript
from .final_contract import canonical_units
from .task_workflow import task_entry
from .utterance_scope import propose_scope, write_scoped_audio


JOB_VERSION = "VSLP_ALIGNMENT_WORKBENCH_JOB_V1"
AUDIO_EXTENSIONS = frozenset({".wav"})
# Engineering review trigger only, not an acoustic word-duration definition.
WORD_SPAN_REVIEW_SEC = 2.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def local_writable_workspace(path: str | Path) -> Path:
    """Accept any explicitly chosen writable workspace without changing sources."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise ValueError("choose_an_absolute_workspace_path")
    target.mkdir(parents=True, exist_ok=True)
    probe = target / f".vslp_write_probe_{uuid4().hex}"
    try:
        probe.write_bytes(b"VSLP")
    finally:
        probe.unlink(missing_ok=True)
    return target


def inspect_audio_files(paths: list[str | Path]) -> list[dict]:
    """Inspect WAV headers without decoding waveforms or hashing entire files."""
    rows: list[dict] = []
    seen: set[Path] = set()
    for source in paths:
        path = Path(source)
        key = path.absolute()
        status = "DUPLICATE" if key in seen else "READY"
        seen.add(key)
        row = {"source_path": str(path), "recording": path.name, "status": status,
               "duration_sec": None, "sample_rate_hz": None, "channels": None,
               "size_bytes": None, "issue": ""}
        if path.suffix.casefold() not in AUDIO_EXTENSIONS:
            row.update(status="UNSUPPORTED FORMAT", issue="wav_only")
        else:
            try:
                info = sf.info(path)
                row.update(duration_sec=float(info.duration),
                           sample_rate_hz=int(info.samplerate),
                           channels=int(info.channels), size_bytes=path.stat().st_size)
                if not info.frames or not info.samplerate:
                    row.update(status="EMPTY", issue="empty_audio")
                elif info.channels > 1 and status == "READY":
                    row.update(status="CHANNEL WARNING", issue="multichannel_audio")
            except (OSError, RuntimeError) as exc:
                row.update(status="UNREADABLE", issue=str(exc)[:200])
        rows.append(row)
    return rows


def discover_audio(folder: str | Path) -> list[Path]:
    """Discover WAV files recursively in stable relative-path order."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError("audio_folder_unavailable")
    return sorted((path for path in root.rglob("*") if path.is_file()
                   and path.suffix.casefold() in AUDIO_EXTENSIONS),
                  key=lambda path: str(path.relative_to(root)).casefold())


def canonical_task(task_id: str, repetitions: int) -> dict:
    task = task_entry(task_id)
    if task is None or not task.get("alignment_applicable") or len(task.get("prompts", [])) != 1:
        raise ValueError("unsupported_alignment_task")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("invalid_expected_repetitions")
    if task_id == "bamboo_passage" and repetitions != 1:
        raise ValueError("bamboo_repetition_contract_mismatch")
    prompt = task["prompts"][0]
    text = str(prompt["exact_expected_text"])
    normalized = normalize_transcript(text)
    return {"task_id": task_id, "task_name": task["display_name"],
            "prompt_id": prompt["prompt_id"], "prompt_version": prompt["prompt_version"],
            "canonical_transcript": text,
            "transcript_hash": sha256(text.encode("utf-8")).hexdigest(),
            "normalized_transcript": normalized,
            "expected_repetitions": repetitions,
            "alignment_transcript": " ".join([normalized] * repetitions)}


def create_job(workspace: str | Path, *, name: str, audio_rows: list[dict],
               task_id: str, repetitions: int) -> Path:
    """Create an isolated job; raw recordings stay at their source paths."""
    ready = [row for row in audio_rows if row["status"] in {"READY", "CHANNEL WARNING"}]
    if not ready:
        raise ValueError("no_readable_audio")
    spec = canonical_task(task_id, repetitions)
    root = local_writable_workspace(workspace) / "alignment_jobs" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("corpus", "mfa_output", "final_alignment", "review", "logs", "exports"):
        (root / folder).mkdir()
    records = []
    for index, row in enumerate(ready, 1):
        records.append({"record_key": f"record_{index:06d}", **row,
                        "state": "QUEUED", "review_status": "UNREVIEWED"})
    manifest = {"job_contract_version": JOB_VERSION, "job_id": root.name,
                "created_at": _now(), "updated_at": _now(), "name": name.strip() or spec["task_name"],
                "state": "CREATED", "input_source": "explicit_audio_selection",
                "task_specification": spec, "profile_path": str(default_profile_path()),
                "recordings": records}
    _write_json(root / "job_manifest.json", manifest)
    return root


def load_job(root: str | Path) -> dict:
    data = json.loads((Path(root) / "job_manifest.json").read_text(encoding="utf-8"))
    if data.get("job_contract_version") != JOB_VERSION:
        raise ValueError("unsupported_alignment_job_version")
    return data


def _save_job(root: Path, job: dict) -> None:
    job["updated_at"] = _now()
    _write_json(root / "job_manifest.json", job)


def prepare_corpus(root: str | Path, progress_callback=None, cancelled=None) -> dict:
    """Write one scoped WAV/LAB pair per record, preserving source audio."""
    root = Path(root)
    job = load_job(root)
    corpus = root / "corpus"
    transcript = job["task_specification"]["alignment_transcript"]
    for index, row in enumerate(job["recordings"], 1):
        if cancelled and cancelled():
            job["state"] = "CANCELLED"
            _save_job(root, job)
            raise AlignmentCancelled("alignment_preparation_cancelled")
        if row["state"] not in {"QUEUED", "CORPUS_READY"}:
            continue
        source = Path(row["source_path"])
        if not source.is_file():
            row.update(state="FAILED", issue="source_audio_missing")
            continue
        speaker = corpus / f"speaker_{index:06d}"
        speaker.mkdir(parents=True, exist_ok=True)
        target = speaker / f"{row['record_key']}.wav"
        scope = row.get("alignment_scope") or propose_scope(source)
        if scope.get("source") == "manual":
            scope = dict(scope)
        start, end = float(scope["start_sec"]), float(scope["end_sec"])
        duration = float(sf.info(source).duration)
        if not 0 <= start < end <= duration + 1e-7:
            raise ValueError("invalid_alignment_scope_bounds")
        if start == 0 and abs(end - duration) < 1 / sf.info(source).samplerate:
            shutil.copy2(source, target)
            start, end = 0.0, duration
            transform = "full_recording_byte_copy"
        else:
            start, end = write_scoped_audio(source, target, start, end)
            transform = "sample_exact_utterance_crop_v1"
        (speaker / f"{row['record_key']}.lab").write_text(transcript, encoding="utf-8")
        row.update(state="CORPUS_READY", corpus_audio=str(target),
                   corpus_transcript=str(speaker / f"{row['record_key']}.lab"),
                   alignment_scope={**scope, "start_sec": start, "end_sec": end},
                   alignment_working_transform=transform)
        if progress_callback:
            progress_callback(index, len(job["recordings"]), f"Prepared {index}/{len(job['recordings'])}")
    issues = validate_corpus_pairs(corpus)
    job["state"] = "CORPUS_READY" if not issues else "PREFLIGHT_FAILED"
    job["corpus_issues"] = issues
    _save_job(root, job)
    return {"status": job["state"], "recordings": len(job["recordings"]),
            "issues": issues, "corpus": str(corpus)}


def set_alignment_scope(root: str | Path, record_key: str, start_sec: float,
                        end_sec: float, reviewer: str) -> dict:
    """Persist a manual utterance scope and invalidate its old MFA result."""
    from .workbench_review import save_corrections

    root = Path(root)
    job = load_job(root)
    row = next((item for item in job["recordings"] if item["record_key"] == record_key), None)
    if row is None or not reviewer.strip():
        raise ValueError("unknown_recording_or_reviewer_missing")
    duration = float(sf.info(row["source_path"]).duration)
    if not 0 <= start_sec < end_sec <= duration + 1e-7:
        raise ValueError("invalid_alignment_scope_bounds")
    save_corrections(root, record_key, {
        "word_bounds": {}, "phone_bounds": {}, "removed_word_indices": [],
        "decision": "", "issue": "Alignment scope changed; re-align required",
    }, reviewer, "Changed alignment utterance scope; invalidated old boundaries")
    row["alignment_scope"] = {"start_sec": start_sec, "end_sec": end_sec,
                              "source": "manual", "reviewer": reviewer.strip(),
                              "uncertain": False, "reason": ""}
    row["textgrid_path"] = ""
    row["state"] = "QUEUED"
    row["review_status"] = "UNREVIEWED"
    job["state"] = "CREATED"
    job.pop("preflight_profile_sha256", None)
    _save_job(root, job)
    return row["alignment_scope"]


def preflight_job(root: str | Path, provider: MfaProvider | None = None) -> dict:
    root = Path(root)
    job = load_job(root)
    if job["state"] != "CORPUS_READY":
        raise ValueError("corpus_not_prepared")
    profile = MfaProfile.load(job["profile_path"])
    provider = provider or MfaProvider()
    result = provider.preflight(root / "corpus", profile,
                                [job["task_specification"]["alignment_transcript"]],
                                root / "logs")
    job["preflight"] = result
    job["preflight_profile_sha256"] = sha256(Path(job["profile_path"]).read_bytes()).hexdigest()
    job["state"] = "READY_TO_ALIGN" if result["status"] == "AVAILABLE" else "PREFLIGHT_FAILED"
    _save_job(root, job)
    return result


def align_job(root: str | Path, provider: MfaProvider | None = None) -> dict:
    root = Path(root)
    job = load_job(root)
    if job["state"] != "READY_TO_ALIGN":
        raise ValueError("preflight_not_ready")
    profile = MfaProfile.load(job["profile_path"])
    provider = provider or MfaProvider()
    environment = job.get("preflight", {})
    current_hash = sha256(Path(job["profile_path"]).read_bytes()).hexdigest()
    if current_hash != job.get("preflight_profile_sha256"):
        raise RuntimeError("mfa_profile_changed_since_preflight")
    provider.configure_from_preflight(profile, environment)
    job["state"] = "RUNNING"
    _save_job(root, job)
    try:
        grids = provider.align_corpus(root / "corpus", root / "mfa_output", profile,
                                      root / "logs")
    except Exception as exc:
        job["state"] = "CANCELLED" if isinstance(exc, AlignmentCancelled) else "FAILED"
        _save_job(root, job)
        raise
    for row in job["recordings"]:
        grid = grids.get(row["record_key"])
        row["textgrid_path"] = str(grid) if grid else ""
        row["state"] = "ALIGNED" if grid else "FAILED"
        row["issue"] = "" if grid else "missing_textgrid"
    job["state"] = "ALIGNED" if all(row["state"] == "ALIGNED" for row in job["recordings"]) else "NEEDS_REVIEW"
    job["mfa_environment"] = environment
    job["mfa_commands"] = provider.commands
    _save_job(root, job)
    qc = qc_job(root)
    return {"status": qc["status"], "aligned": sum(bool(row["textgrid_path"])
                                                 for row in job["recordings"]),
            "total": len(job["recordings"]), "qc": qc}


def realign_recording(root: str | Path, record_key: str, spoken_transcript: str,
                      reviewer: str, provider: MfaProvider | None = None) -> dict:
    """Re-run MFA for one recording after a lexical edit; preserve the old grid."""
    from .workbench_review import save_corrections

    root = Path(root)
    job = load_job(root)
    row = next((item for item in job["recordings"] if item["record_key"] == record_key), None)
    if row is None:
        raise ValueError("unknown_alignment_recording")
    if not reviewer.strip():
        raise ValueError("reviewer_required")
    normalized = normalize_transcript(spoken_transcript)
    if not normalized:
        raise ValueError("spoken_transcript_empty")
    source = Path(row["source_path"])
    if not source.is_file():
        raise ValueError("source_audio_missing")
    attempt = root / "realign" / record_key / uuid4().hex
    speaker = attempt / "corpus" / "speaker_000001"
    speaker.mkdir(parents=True)
    scope = row.get("alignment_scope", {})
    start, end = float(scope.get("start_sec", 0)), float(scope.get("end_sec", sf.info(source).duration))
    if start == 0 and abs(end - sf.info(source).duration) < 1 / sf.info(source).samplerate:
        shutil.copy2(source, speaker / f"{record_key}.wav")
    else:
        write_scoped_audio(source, speaker / f"{record_key}.wav", start, end)
    (speaker / f"{record_key}.lab").write_text(normalized, encoding="utf-8")
    profile = MfaProfile.load(job["profile_path"])
    provider = provider or MfaProvider()
    preflight = provider.preflight(attempt / "corpus", profile, [normalized],
                                   attempt / "logs")
    if preflight["status"] != "AVAILABLE":
        raise ValueError(f"spoken_transcript_preflight_failed:{preflight['status']}")
    provider.configure_from_preflight(profile, preflight)
    grids = provider.align_corpus(attempt / "corpus", attempt / "mfa_output", profile,
                                  attempt / "logs")
    grid = grids.get(record_key)
    if grid is None:
        raise ValueError("realignment_textgrid_missing")
    # A lexical edit invalidates old word/phone corrections. The previous
    # TextGrid and correction history remain on disk for audit and rollback.
    save_corrections(root, record_key, {
        "reviewed_spoken_transcript": spoken_transcript.strip(),
        "realignment_textgrid_path": str(grid), "decision": "",
        "removed_word_indices": [], "word_bounds": {}, "phone_bounds": {},
        "issue": "Transcript mismatch"}, reviewer, "Re-aligned reviewed spoken transcript")
    row["textgrid_path"] = str(grid)
    row["state"] = "ALIGNED"
    row["realignment_attempt"] = str(attempt)
    _save_job(root, job)
    return qc_job(root)


def qc_job(root: str | Path) -> dict:
    """Structural QC only; no unpublished duration/quality thresholds."""
    root = Path(root)
    from .workbench_review import load_corrections, reviewed_tokens

    job = load_job(root)
    expected = job["task_specification"]["alignment_transcript"].split()
    protocol_count = len(canonical_units(job["task_specification"]["canonical_transcript"]))
    results = []
    for row in job["recordings"]:
        flags = []
        scope = row.get("alignment_scope", {})
        scope_start = float(scope.get("start_sec", 0))
        scope_end = float(scope.get("end_sec", row["duration_sec"]))
        if scope.get("uncertain"):
            flags.append("alignment_scope_requires_review")
        if scope and scope_end - scope_start < len(expected) * .05:
            flags.append("transcript_cannot_fit_proposed_scope")
        if scope and (scope_start < .05 or float(row["duration_sec"]) - scope_end < .05):
            flags.append("alignment_scope_touches_recording_boundary")
        if not row.get("textgrid_path"):
            flags.append("mfa_output_missing")
            word_count = phone_count = 0
            first_word_start = None
            extra_tokens = []
            long_word_spans = []
        else:
            try:
                words, phones = reviewed_tokens(root, row["record_key"])
                actual = [normalize_transcript(str(label)) for label in words.get("label", [])]
                first_word_start = float(words.start_sec.iloc[0]) if not words.empty else None
                if scope and not words.empty and (first_word_start - scope_start > 2 or
                                        scope_end - float(words.end_sec.iloc[-1]) > 2):
                    flags.append("large_unaligned_scope_region_requires_review")
                if scope and first_word_start is not None and first_word_start - scope_start > 2:
                    flags.append("leading_silence_too_long")
                long_word_spans = [
                    {"label": str(word.label), "start_sec": float(word.start_sec),
                     "end_sec": float(word.end_sec)}
                    for word in words.itertuples()
                    if float(word.end_sec) - float(word.start_sec) > WORD_SPAN_REVIEW_SEC]
                if long_word_spans:
                    flags.append("long_aligned_word_span_requires_review")
                    if float(words.end_sec.iloc[0]) - first_word_start > WORD_SPAN_REVIEW_SEC:
                        flags.append("first_lexical_word_implausibly_long")
                    if (float(words.end_sec.iloc[-1]) - float(words.start_sec.iloc[-1]) >
                            WORD_SPAN_REVIEW_SEC):
                        flags.append("last_lexical_word_implausibly_long")
                from difflib import SequenceMatcher

                extra_tokens = [actual[index] for op, _start, _end, begin, finish in
                                SequenceMatcher(a=expected, b=actual, autojunk=False).get_opcodes()
                                if op in {"insert", "replace"}
                                for index in range(begin, finish)]
                corrections = load_corrections(root, row["record_key"])
                spoken = corrections.get("reviewed_spoken_transcript", "")
                reviewed_expected = normalize_transcript(spoken).split() if spoken else expected
                if spoken and reviewed_expected != expected:
                    flags.append("spoken_transcript_differs_from_canonical")
                word_count, phone_count = len(words), len(phones)
                if actual != reviewed_expected:
                    flags.append("reviewed_transcript_alignment_mismatch" if spoken else
                                 "transcript_token_mismatch")
                if words.empty or phones.empty:
                    flags.append("empty_alignment_tier")
                for table in (words, phones):
                    if not table.empty:
                        starts = table.start_sec.astype(float).to_numpy()
                        ends = table.end_sec.astype(float).to_numpy()
                        if (not (starts < ends).all() or
                                not (starts[1:] >= starts[:-1]).all() or
                                not (starts[1:] >= ends[:-1] - 1e-7).all() or
                                float(ends[-1]) > float(row["duration_sec"]) + 1e-7):
                            flags.append("invalid_alignment_bounds")
                        if (float(starts[0]) < scope_start - 1e-7 or
                                float(ends[-1]) > scope_end + 1e-7):
                            flags.append("alignment_outside_utterance_scope")
            except (OSError, ValueError, KeyError) as exc:
                word_count = phone_count = 0
                first_word_start = None
                extra_tokens = []
                long_word_spans = []
                flags.append(f"textgrid_parse_failed:{exc}")
        review = load_corrections(root, row["record_key"])
        decision = review.get("decision", "")
        status = ("EXCLUDED" if decision == "EXCLUDE" else
                  "REVIEWED" if decision == "ACCEPT" and not flags else
                  "NEEDS_REVIEW" if flags or decision == "FLAG" else "AUTO_PASS")
        row.update(state=status, qc_flags=flags, aligned_word_count=word_count,
                   aligned_phone_count=phone_count,
                   expected_provider_word_count=len(expected),
                   protocol_scientific_word_count=protocol_count,
                   first_aligned_word_start_sec=first_word_start,
                   extra_aligned_tokens=extra_tokens,
                   long_aligned_word_spans=long_word_spans)
        results.append({"record_key": row["record_key"], "recording": row["recording"],
                        "status": status, "word_count": word_count,
                        "phone_count": phone_count, "expected_word_count": len(expected),
                        "protocol_word_count": protocol_count,
                        "first_word_start_sec": first_word_start,
                        "extra_aligned_tokens": extra_tokens,
                        "long_aligned_word_spans": long_word_spans,
                        "flags": flags})
    job["state"] = ("NEEDS_REVIEW" if any(row["status"] == "NEEDS_REVIEW" for row in results)
                    else "REVIEWED" if any(row["status"] == "REVIEWED" for row in results)
                    else "AUTO_PASS")
    _save_job(root, job)
    _write_json(root / "review" / "qc_summary.json", {"job_id": job["job_id"],
                                                   "records": results})
    return {"status": job["state"], "auto_pass": sum(row["status"] == "AUTO_PASS"
                                                     for row in results),
            "reviewed": sum(row["status"] == "REVIEWED" for row in results),
            "needs_review": sum(row["status"] == "NEEDS_REVIEW" for row in results),
            "records": results}
