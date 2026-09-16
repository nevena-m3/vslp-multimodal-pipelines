import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vslp.acoustic.run_setup import initialize_acoustic_run
from vslp.core.project import prune_empty_acoustic_directories, task_run_folder_name


WHEN = datetime(2026, 9, 16, 14, 5, 9, 123456, tzinfo=timezone(timedelta(hours=-4)))


def test_task_run_folder_name_is_required_and_timestamped():
    assert task_run_folder_name("Bamboo passage", WHEN) == "Bamboo_passage_20260916_140509"
    with pytest.raises(ValueError):
        task_run_folder_name(" / ", WHEN)


def test_setup_creates_only_acoustic_run_with_exact_values(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    parent = tmp_path / "new" / "output"
    raw = {"project_name": "  Speech Study  ", "task_name": "Bamboo passage /a/", "input_folder": str(source), "output_parent": str(parent)}
    run = initialize_acoustic_run(project_name=raw["project_name"], task_name=raw["task_name"], input_folder=source, output_parent=parent, setup_values=raw, now=WHEN)
    assert run.run_id == "Bamboo_passage_a_20260916_140509"
    assert {path.name for path in run.root.iterdir()} == {"project_manifest.json", "configs", "logs", "acoustic"}
    manifest = json.loads((run.root / "project_manifest.json").read_text(encoding="utf-8"))
    assert manifest["project_name"] == "Speech Study"
    assert manifest["task_name"] == "Bamboo passage /a/"
    assert manifest["task_slug"] == "Bamboo_passage_a"
    assert manifest["created_at_utc"] == "2026-09-16T18:05:09.123456+00:00"
    assert manifest["run_root"] == str(run.root)
    assert manifest["input_folder"] == str(source)
    assert manifest["output_parent"] == str(parent)
    assert manifest["modality"] == "acoustic"
    assert manifest["gui_version"] and manifest["pipeline_version"]
    assert json.loads((run.root / "configs" / "setup_config.json").read_text(encoding="utf-8")) == raw
    assert (run.root / "logs" / "setup.log").is_file()


def test_collision_never_reuses_existing_run(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    parent = tmp_path / "output"
    parent.mkdir()
    existing = parent / "Task_20260916_140509"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    kwargs = dict(project_name="Study", task_name="Task", input_folder=source, output_parent=parent, setup_values={}, now=WHEN)
    first = initialize_acoustic_run(**kwargs)
    second = initialize_acoustic_run(**kwargs)
    assert first.run_id == "Task_20260916_140509_02"
    assert second.run_id == "Task_20260916_140509_03"
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_setup_rejects_invalid_paths_and_names(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    base = dict(project_name="Study", task_name="Task", input_folder=source, output_parent=tmp_path / "output", setup_values={}, now=WHEN)
    for changed in ({"project_name": " "}, {"task_name": " "}, {"input_folder": tmp_path / "missing"}, {"output_parent": source / "inside"}):
        with pytest.raises(ValueError):
            initialize_acoustic_run(**(base | changed))


def test_prune_empty_stage_folders_preserves_acoustic_root(tmp_path: Path):
    component = tmp_path / "acoustic"
    empty = component / "001_ingest" / "plots"
    empty.mkdir(parents=True)
    prune_empty_acoustic_directories(tmp_path)
    assert component.is_dir()
    assert not empty.exists()
