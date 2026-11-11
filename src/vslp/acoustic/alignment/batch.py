"""One-time, whole-dataset MFA alignment outside the Acoustic GUI.

Runs the standalone Alignment workbench (the same code the VSLP Alignment app
uses) over every recording of each fixed-prompt task, then exports one
``VSLP_ALIGNMENT_PACKAGE_V1`` per task. Any Acoustic project can import that
package; recordings are matched by decoded audio content, not by file name.

Usage (from the vslp-app environment; MFA lives in its own conda env)::

    python -m vslp.acoustic.alignment.batch \
        --inventory "...\\WAVS.csv" --zip "...\\WAVs-....zip" \
        --workspace "C:\\VSLP_alignment_batch" --limit-per-task 10

Source recordings are never modified. Only needed WAVs are extracted from the
zip into ``<workspace>/audio``. Re-running skips finished tasks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import pandas as pd
import yaml

from .workbench import (align_job, create_job, inspect_audio_files, load_job,
                        preflight_job, prepare_corpus)
from .workbench_package import export_standalone_package

TASKS_PATH = Path(__file__).with_name("batch_tasks.yaml")
# Whole-dataset default: same model/dictionary as the engineering profile, but a
# wide Viterbi beam (100/400) so slow, dysarthric speech is not silently dropped.
BATCH_PROFILE = Path(__file__).parent / "profiles" / "mfa_vslp_en_us_arpa_334_wide_beam_v1.json"
NAME_PATTERN = re.compile(
    r"^(?P<subject>[^_]+)_(?P<protocol>\d+)_(?P<iteration>\d+)_(?P<date>\d{8})_(?P<code>.+)$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_task_table(path: str | Path = TASKS_PATH) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != "vslp_batch_alignment_tasks_v1":
        raise ValueError("unsupported_batch_task_table")
    return data


def parse_file_name(name: str) -> dict | None:
    """Split SUBJECT_PROTOCOL_ITERATION_YYYYMMDD_CODE.wav; None if it does not fit."""
    match = NAME_PATTERN.match(Path(name).stem)
    return match.groupdict() if match else None


def build_inventory(inventory_csv: str | Path | None, names: list[str] | None = None,
                    task_table: dict | None = None) -> pd.DataFrame:
    """One row per WAV with subject, task code and the batch task it maps to."""
    table = task_table or load_task_table()
    if inventory_csv:
        raw = pd.read_csv(inventory_csv, dtype=str, keep_default_na=False)
        column = "Raw Media File name"
        if column not in raw:
            raise ValueError("inventory_missing_file_name_column")
        names = raw[column].tolist()
        subjects = dict(zip(raw[column], raw.get("SubjectID", pd.Series([""] * len(raw)))))
    else:
        subjects = {}
    rows = []
    for name in names or []:
        if not str(name).casefold().endswith(".wav"):
            continue
        parts = parse_file_name(name)
        code = parts["code"] if parts else ""
        spec = table["tasks"].get(code)
        rows.append({"file_name": name, "subject_id": subjects.get(name) or
                     (parts["subject"] if parts else ""), "task_code": code,
                     "task_id": spec["task_id"] if spec else "",
                     "repetitions": int(spec["repetitions"]) if spec else 0,
                     "batch_role": ("fixed_prompt" if spec else
                                    "identify_first" if code in table.get("identify_first", {})
                                    else "not_aligned" if code in table.get("not_aligned", [])
                                    else "not_in_batch")})
    return pd.DataFrame(rows, columns=["file_name", "subject_id", "task_code", "task_id",
                                       "repetitions", "batch_role"])


def extract_from_zip(zip_path: str | Path, names: list[str], destination: str | Path) -> dict:
    """Extract only the named WAVs (by base name) into a flat folder; skip finished ones."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    wanted = set(names)
    found: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            base = Path(info.filename).name
            if info.is_dir() or base not in wanted or base in found:
                continue
            target = destination / base
            if not (target.is_file() and target.stat().st_size == info.file_size):
                temporary = target.with_suffix(".part")
                with archive.open(info) as source, temporary.open("wb") as sink:
                    shutil.copyfileobj(source, sink, 1024 * 1024)
                temporary.replace(target)
            found[base] = target
    return {"found": found, "missing": sorted(wanted - set(found))}


def locate_audio(audio_dir: str | Path, names: list[str]) -> dict:
    index = {path.name: path for path in Path(audio_dir).rglob("*.wav")}
    found = {name: index[name] for name in names if name in index}
    return {"found": found, "missing": sorted(set(names) - set(found))}


def _speaker_id(subject: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", subject.strip())


def run_task(job_name: str, rows: pd.DataFrame, paths: dict[str, Path], workspace: Path,
             *, profile_path: str = "", provider=None, log=print) -> dict:
    """Align one fixed-prompt task code and export its package."""
    task_id, repetitions = str(rows.task_id.iloc[0]), int(rows.repetitions.iloc[0])
    audio_rows = inspect_audio_files([paths[name] for name in rows.file_name])
    subject_by_name = dict(zip(rows.file_name, rows.subject_id))
    for row in audio_rows:
        row["speaker_id"] = _speaker_id(subject_by_name.get(row["recording"], ""))
    job_root = create_job(workspace / "jobs", name=job_name, audio_rows=audio_rows,
                          task_id=task_id, repetitions=repetitions)
    if profile_path:
        job = load_job(job_root)
        job["profile_path"] = str(Path(profile_path).resolve())
        (job_root / "job_manifest.json").write_text(json.dumps(job, indent=2),
                                                    encoding="utf-8")
    log(f"[{job_name}] {len(audio_rows)} recordings -> job {job_root.name}")
    prepared = prepare_corpus(job_root)
    if prepared["status"] != "CORPUS_READY":
        return {"job": job_name, "status": "CORPUS_FAILED", "job_root": str(job_root),
                "detail": prepared}
    preflight = preflight_job(job_root, provider)
    if preflight["status"] != "AVAILABLE":
        return {"job": job_name, "status": f"PREFLIGHT_{preflight['status']}",
                "job_root": str(job_root), "oov_words": preflight.get("oov_words", [])}
    log(f"[{job_name}] MFA validation passed; aligning (one corpus-level MFA run)...")
    aligned = align_job(job_root, provider)
    package = workspace / "packages" / job_name
    if (package / "package_manifest.json").is_file():
        shutil.rmtree(package)  # stale package from an earlier run of this task
    exported = 0
    if aligned["qc"]["auto_pass"]:
        export_standalone_package(job_root, package, skip_unmappable=True)
        exported = int(json.loads((package / "package_manifest.json").read_text(
            encoding="utf-8"))["recording_count"])
    log(f"[{job_name}] auto-pass {aligned['qc']['auto_pass']}, needs review "
        f"{aligned['qc']['needs_review']}, exported {exported}")
    return {"job": job_name, "status": "DONE", "job_root": str(job_root),
            "package": str(package) if exported else "", "recordings": len(audio_rows),
            "auto_pass": aligned["qc"]["auto_pass"],
            "needs_review": aligned["qc"]["needs_review"], "exported": exported}


def recording_table(job_root: str | Path, job_name: str) -> pd.DataFrame:
    job = load_job(job_root)
    return pd.DataFrame([{
        "job": job_name, "file_name": row["recording"], "record_key": row["record_key"],
        "speaker_id": row.get("speaker_id", ""), "state": row["state"],
        "qc_flags": ";".join(row.get("qc_flags", [])),
        "aligned_words": row.get("aligned_word_count", 0),
        "expected_words": row.get("expected_provider_word_count", 0),
        "first_word_start_sec": row.get("first_aligned_word_start_sec"),
        "textgrid_path": row.get("textgrid_path", ""), "source_path": row["source_path"],
    } for row in job["recordings"]])


def plot_recording(job_root: str | Path, record_key: str, source: str | Path,
                   destination: str | Path, title: str) -> Path:
    """Waveform + spectrogram + word and phone tiers in original recording time.

    Tiers come from the job exactly as the package exports them (MFA times
    shifted by the utterance crop), so the picture shows what the GUI imports.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    from .workbench_review import reviewed_tokens

    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    try:
        words, phones = reviewed_tokens(job_root, record_key)
    except ValueError:  # MFA produced nothing for this recording
        words = phones = pd.DataFrame(columns=["label", "start_sec", "end_sec"])
    duration = len(mono) / rate
    figure, axes = plt.subplots(4, 1, figsize=(14, 7), sharex=True,
                                gridspec_kw={"height_ratios": [1.2, 2, .6, .6]})
    time = np.arange(len(mono)) / rate
    step = max(1, len(mono) // 200000)
    axes[0].plot(time[::step], mono[::step], color="#555", lw=.4)
    axes[0].set_yticks([])
    with np.errstate(divide="ignore"):  # digital silence has zero power
        _spec, _freqs, _bins, image = axes[1].specgram(mono, NFFT=512, Fs=rate,
                                                       noverlap=384, cmap="magma")
    finite = image.get_array()[np.isfinite(image.get_array())]
    if finite.size:  # 70 dB display range so silence reads as dark, not blank
        image.set_clim(float(finite.max()) - 70, float(finite.max()))
    axes[1].set_facecolor("black")
    axes[1].set_ylim(0, 8000)
    axes[1].set_ylabel("Hz")
    for axis, table, color, size in ((axes[2], words, "#2a6fdb", 8),
                                     (axes[3], phones, "#2e9d5b", 6)):
        axis.set_yticks([])
        for token in table.itertuples():
            start, end = float(token.start_sec), float(token.end_sec)
            axis.axvspan(start, end, color=color, alpha=.18)
            axis.axvline(start, color=color, lw=.6)
            axis.text((start + end) / 2, .5, str(token.label), ha="center", va="center",
                      fontsize=size, clip_on=True)
    axes[2].set_ylabel("words", rotation=0, ha="right")
    axes[3].set_ylabel("phones", rotation=0, ha="right")
    axes[3].set_xlim(0, duration)
    axes[3].set_xlabel("original recording time (s)")
    axes[0].set_title(title, fontsize=10, loc="left")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return destination


def run_batch(*, inventory_csv: str | None, workspace: str | Path, zip_path: str | None = None,
              audio_dir: str | None = None, codes: list[str] | None = None,
              limit_per_task: int = 0, profile_path: str = "", provider=None,
              plot: str = "flagged", log=print) -> dict:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    table = load_task_table()
    names = None
    if not inventory_csv:
        if audio_dir:
            names = [path.name for path in Path(audio_dir).rglob("*.wav")]
        elif zip_path:
            with zipfile.ZipFile(zip_path) as archive:
                names = [Path(item).name for item in archive.namelist()]
    inventory = build_inventory(inventory_csv, names, table)
    inventory.to_csv(workspace / "inventory_with_batch_roles.csv", index=False)
    fixed = inventory.loc[inventory.batch_role.eq("fixed_prompt")]
    if codes:
        fixed = fixed.loc[fixed.task_code.isin(codes)]
    if limit_per_task:
        # One recording per participant, at most N per task (pandas 2 and 3 safe).
        fixed = (fixed.sort_values(["task_code", "subject_id", "file_name"])
                 .drop_duplicates(["task_code", "subject_id"])
                 .groupby("task_code").head(limit_per_task))
    names_needed = fixed.file_name.tolist()
    log(f"{len(names_needed)} recordings in {fixed.task_code.nunique()} task codes")
    if audio_dir:
        located = locate_audio(audio_dir, names_needed)
    elif zip_path:
        log("Extracting needed WAVs from the zip (read only)...")
        located = extract_from_zip(zip_path, names_needed, workspace / "audio")
    else:
        raise ValueError("give --zip or --audio-dir")
    if located["missing"]:
        log(f"WARNING: {len(located['missing'])} inventory files not found in the audio source")
        (workspace / "missing_audio.txt").write_text("\n".join(located["missing"]),
                                                    encoding="utf-8")
    state_path = workspace / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "created_utc": _now(), "jobs": {}}
    tables = []
    for code, rows in fixed.groupby("task_code"):
        rows = rows.loc[rows.file_name.isin(located["found"])]
        job_name = f"{code}__{rows.task_id.iloc[0]}" if len(rows) else code
        if rows.empty:
            continue
        previous = state["jobs"].get(job_name)
        if previous and previous.get("status") == "DONE" and previous.get(
                "recordings") == len(rows):
            log(f"[{job_name}] already done; skipping")
            result = previous
        else:
            try:
                result = run_task(job_name, rows, located["found"], workspace,
                                  profile_path=profile_path, provider=provider, log=log)
            except Exception as exc:  # one failed task must not stop the others
                result = {"job": job_name, "status": "FAILED", "error": repr(exc)[:500]}
                log(f"[{job_name}] FAILED: {exc}")
            state["jobs"][job_name] = {**result, "finished_utc": _now()}
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if result.get("job_root"):
            records = recording_table(result["job_root"], job_name)
            tables.append(records)
            if plot != "none":
                chosen = records if plot == "all" else records.loc[records.state.ne("AUTO_PASS")]
                for item in chosen.itertuples():
                    target = workspace / "plots" / job_name / f"{Path(item.file_name).stem}.png"
                    if not target.is_file():
                        try:
                            plot_recording(result["job_root"], item.record_key,
                                           item.source_path, target,
                                           f"{item.file_name} · {item.state} · {item.qc_flags}")
                        except Exception as exc:  # noqa: BLE001 - plots never block results
                            log(f"plot failed for {item.file_name}: {exc}")
    summary = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    summary.to_csv(workspace / "batch_recordings.csv", index=False)
    log(f"Done. Packages in {workspace / 'packages'}; per-recording table: "
        f"{workspace / 'batch_recordings.csv'}")
    return {"workspace": str(workspace), "jobs": state["jobs"], "recordings": len(summary)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inventory", help="WAVS.csv inventory (optional)")
    parser.add_argument("--zip", help="WAV zip archive (read only)")
    parser.add_argument("--audio-dir", help="folder that already holds the WAVs")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--codes", nargs="*", help="only these task codes, e.g. 240_PSG_BAMBOO")
    parser.add_argument("--limit-per-task", type=int, default=0,
                        help="pilot: at most N recordings per task (one per participant)")
    parser.add_argument("--profile", default=str(BATCH_PROFILE),
                        help="MFA profile JSON (default: wide-beam batch profile)")
    parser.add_argument("--plots", choices=["flagged", "all", "none"], default="flagged")
    args = parser.parse_args(argv)
    result = run_batch(inventory_csv=args.inventory, workspace=args.workspace,
                       zip_path=args.zip, audio_dir=args.audio_dir, codes=args.codes,
                       limit_per_task=args.limit_per_task, profile_path=args.profile,
                       plot=args.plots)
    failed = [name for name, job in result["jobs"].items() if job.get("status") != "DONE"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
