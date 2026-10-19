"""Package interchange preserves the frozen Final Alignment feature contract."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import struct

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from tests.fixtures.final_alignment_contract_v1 import write_nonclinical_final_alignment
from vslp.acoustic.alignment import load_final_alignment
from vslp.acoustic.alignment.audio_identity import (
    audio_content_sha256, legacy_peak_timestamp_only_match,
)
from vslp.acoustic.alignment.package_v1 import (
    export_alignment_package, import_alignment_package, inspect_alignment_package,
    validate_alignment_package,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared(tmp_path):
    root = write_nonclinical_final_alignment(tmp_path / "project", "bamboo_passage")
    review = root / "acoustic" / "003_segmentation_review" / "final"
    audio = tmp_path / "synthetic.wav"
    sf.write(audio, np.zeros(30 * 16000, dtype=np.float32), 16000, subtype="FLOAT")
    decisions_path = review / "final_segmentation_decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions["analysis_wav_path"] = str(audio)
    decisions["file_name"] = audio.name
    decisions.to_csv(decisions_path, index=False)
    final = root / "acoustic" / "004_alignment" / "final"
    manifest_path = final / "final_alignment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_final_decisions_sha256"] = _hash(decisions_path)
    manifest["source_audio_sha256_by_recording"] = {"nonclinical": _hash(audio)}
    manifest["source_audio_content_sha256_by_recording"] = {
        "nonclinical": audio_content_sha256(audio)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    package = export_alignment_package(root, tmp_path / "package")
    return root, package


def _rewrite(package: Path, name: str, table: pd.DataFrame) -> None:
    path = package / name
    table.to_csv(path, index=False)
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["package_file_hashes"][name] = _hash(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_export_import_retains_canonical_tables_and_loader_precedence(prepared):
    root, package = prepared
    embedded, issue = load_final_alignment(root)
    assert issue == ""
    result = validate_alignment_package(package, root)
    assert result["matched"] == result["finalized"] == 1
    assert result["manifest"]["recording_count"] == 1
    installed = import_alignment_package(package, root)
    assert installed.parent == root / "alignment_imports"
    imported, issue = load_final_alignment(root)
    assert issue == ""
    for name in ("trials", "canonical_words", "canonical_phones", "words", "phones"):
        pd.testing.assert_frame_equal(getattr(embedded, name), getattr(imported, name),
                                      check_exact=True)
    assert imported.manifest["alignment_run_id"] == embedded.manifest["alignment_run_id"]
    assert imported.phones_for_trial("nonclinical", "trial_001").equals(
        embedded.phones_for_trial("nonclinical", "trial_001"))
    pd.testing.assert_frame_equal(embedded.family07_rows(), imported.family07_rows(),
                                  check_exact=True)
    pd.testing.assert_frame_equal(embedded.get_vowel_tokens("nonclinical"),
                                  imported.get_vowel_tokens("nonclinical"), check_exact=True)
    (installed / "final_alignment_canonical_words.csv").write_bytes(b"tampered")
    missing, issue = load_final_alignment(root)
    assert missing is None
    assert "package_file_hash_mismatch" in issue


def test_acoustic_gui_imports_explicit_package_without_embedded_alignment_controls(
        prepared, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog

    from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow

    root, package = prepared
    app = QApplication.instance() or QApplication([])
    window = AcousticPipelineWindow()
    window._run_root = root
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(package))
    window._select_alignment_package()
    assert "Matched 1/1" in window.feature_package_status.text()
    assert window.alignment_widget is None
    assert "Alignment" not in [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert load_final_alignment(root)[0] is not None
    window.close()
    assert app is not None


def test_import_into_project_without_embedded_final_alignment(prepared, tmp_path):
    root, package = prepared
    isolated = tmp_path / "isolated_project"
    source_review = root / "acoustic" / "003_segmentation_review" / "final"
    target_review = isolated / "acoustic" / "003_segmentation_review" / "final"
    target_review.mkdir(parents=True)
    for name in ("final_segmentation_decisions.csv", "final_segmentation_intervals.csv"):
        shutil.copy2(source_review / name, target_review / name)
    assert load_final_alignment(isolated)[1] == "missing_alignment"
    import_alignment_package(package, isolated)
    loaded, issue = load_final_alignment(isolated)
    assert issue == ""
    assert loaded is not None
    assert len(loaded.canonical_words) == 97


def test_container_metadata_does_not_change_audio_identity(prepared):
    root, package = prepared
    decisions = pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" /
                            "final_segmentation_decisions.csv")
    audio = Path(decisions.analysis_wav_path.iloc[0])
    before = audio_content_sha256(audio)
    payload = audio.read_bytes()
    extra = b"LIST" + struct.pack("<I", 4) + b"test"
    changed = payload + extra
    audio.write_bytes(changed[:4] + struct.pack("<I", len(changed) - 8) + changed[8:])
    assert audio_content_sha256(audio) == before
    assert validate_alignment_package(package, root)["matched"] == 1
    preview = inspect_alignment_package(package, root)
    assert preview["stale"] == 0
    assert preview["recording_status"][0]["status"] == "MATCHED + FINALIZED"


def test_peak_timestamp_only_change_preserves_content_and_exact_legacy_proof(prepared):
    root, package = prepared
    decisions = pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" /
                            "final_segmentation_decisions.csv")
    audio = Path(decisions.analysis_wav_path.iloc[0])
    payload = audio.read_bytes()
    peak = payload.index(b"PEAK")
    original = 1_800_000_000
    audio.write_bytes(payload[:peak + 12] + struct.pack("<I", original) +
                      payload[peak + 16:])
    original_hash = _hash(audio)
    original_content = audio_content_sha256(audio)
    payload = audio.read_bytes()
    audio.write_bytes(payload[:peak + 12] + struct.pack("<I", original + 20) +
                      payload[peak + 16:])
    assert audio_content_sha256(audio) == original_content
    assert legacy_peak_timestamp_only_match(audio, original_hash,
        "2020-01-15T08:00:00+00:00", "2020-01-15T08:10:00+00:00") is False
    from datetime import datetime, timezone
    lower = datetime.fromtimestamp(original - 1, timezone.utc).isoformat()
    upper = datetime.fromtimestamp(original + 1, timezone.utc).isoformat()
    assert legacy_peak_timestamp_only_match(audio, original_hash, lower, upper)
    assert validate_alignment_package(package, root)["matched"] == 1
    samples, rate = sf.read(audio)
    samples[100] = .5
    sf.write(audio, samples, rate, subtype="FLOAT")
    assert not legacy_peak_timestamp_only_match(audio, original_hash, lower, upper)


@pytest.mark.parametrize("change", ["sample", "rate", "channels", "count"])
def test_signal_identity_change_invalidates_alignment_and_package(prepared, change):
    root, package = prepared
    decisions = pd.read_csv(root / "acoustic" / "003_segmentation_review" / "final" /
                            "final_segmentation_decisions.csv")
    audio = Path(decisions.analysis_wav_path.iloc[0])
    samples, rate = sf.read(audio, dtype="float64")
    if change == "sample":
        samples[100] = .5
    elif change == "rate":
        rate += 1
    elif change == "channels":
        samples = np.column_stack((samples, samples))
    else:
        samples = samples[:-1]
    sf.write(audio, samples, rate)
    assert load_final_alignment(root)[1] == "alignment_source_audio_changed"
    with pytest.raises(ValueError, match="IDENTITY MISMATCH"):
        validate_alignment_package(package, root)


@pytest.mark.parametrize("mutation,reason", [
    ("version", "unsupported_package_version"),
    ("missing", "package_file_inventory_mismatch"),
    ("hash", "package_file_hash_mismatch"),
    ("duplicate", "duplicate_or_missing_record_key"),
    ("task", "TASK/PROMPT MISMATCH"),
    ("prompt", "TASK/PROMPT MISMATCH"),
    ("audio", "IDENTITY MISMATCH"),
    ("nonfinal", "alignment_not_finalized"),
    ("trial_bounds", "invalid_trial_bounds"),
    ("word_bounds", "token_outside_trial"),
    ("phone_bounds", "phone_outside_parent_word"),
])
def test_invalid_packages_are_rejected(prepared, tmp_path, mutation, reason):
    root, source = prepared
    package = tmp_path / "tampered"
    shutil.copytree(source, package)
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "version":
        manifest["package_contract_version"] = "future"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "missing":
        (package / "final_alignment_canonical_words.csv").unlink()
    elif mutation == "hash":
        with (package / "final_alignment_canonical_words.csv").open("ab") as output:
            output.write(b"tampered")
    else:
        name = ("recording_status.csv" if mutation in {
            "duplicate", "task", "prompt", "audio", "nonfinal"}
                else "final_alignment_contract_trials.csv" if mutation == "trial_bounds"
                else "final_alignment_canonical_words.csv" if mutation == "word_bounds"
                else "final_alignment_canonical_phones.csv")
        table = pd.read_csv(package / name, keep_default_na=False)
        if mutation == "duplicate":
            table = pd.concat([table, table], ignore_index=True)
        elif mutation == "task":
            table.loc[0, "task_id"] = "sentence_s3"
        elif mutation == "prompt":
            table.loc[0, "transcript_hash"] = "wrong"
        elif mutation == "audio":
            table.loc[0, "source_audio_content_sha256"] = "wrong"
        elif mutation == "nonfinal":
            table.loc[0, "finalization_status"] = "ALIGNED"
        elif mutation == "trial_bounds":
            table.loc[0, "end_sec"] = table.loc[0, "start_sec"]
        elif mutation == "word_bounds":
            table.loc[0, "start_sec"] = 1000.
        else:
            table.loc[0, "end_sec"] = float(table.loc[0, "end_sec"]) + .1
        _rewrite(package, name, table)
    with pytest.raises(ValueError, match=reason):
        validate_alignment_package(package, root)
