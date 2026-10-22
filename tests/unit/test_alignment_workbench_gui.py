"""The launched Alignment app exposes the task-first batch workbench."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication
import soundfile as sf

from vslp.acoustic.alignment.workbench import create_job, inspect_audio_files
from vslp.acoustic.alignment.workbench import load_job, qc_job
from vslp.acoustic.alignment.workbench_review import load_corrections
from vslp.gui.alignment_app.workbench_review import WorkbenchReview, load_recording_review
from vslp.gui.alignment_app.workbench_window import WorkbenchWindow


def test_workbench_ingest_task_batch_and_review_queue(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(WorkbenchWindow, "_start", lambda *_args, **_kwargs: None)
    window = WorkbenchWindow()
    try:
        assert window.windowTitle() == "VSLP Alignment"
        assert window.tabs.tabText(0) == "1  Job setup"
        assert window.task_type.currentData() == "bamboo_passage"
        assert window.repetitions.value() == 1
        assert "Bamboo walls" in window.transcript.toPlainText()
        audio = tmp_path / "bamboo.wav"
        sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
        window.audio_rows = inspect_audio_files([audio])
        window._show_ingest()
        assert window.ingest_table.rowCount() == 1
        assert "1 ready" in window.ingest_summary.text()
        window.job_root = create_job(tmp_path / "workspace", name="Bamboo test",
                                     audio_rows=window.audio_rows,
                                     task_id="bamboo_passage", repetitions=1)
        window._show_job()
        assert window.batch_table.rowCount() == 1
        assert window.review_table.rowCount() == 1
        assert "Bamboo test" in window.job_heading.text()
        assert window.log_dock.widget() is not None
        app.processEvents()
    finally:
        window.close()


def test_review_panel_trial_playback_and_correction_provenance(tmp_path):
    app = QApplication.instance() or QApplication([])
    audio = tmp_path / "s2.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    root = create_job(tmp_path / "workspace", name="S2", audio_rows=inspect_audio_files([audio]),
                      task_id="sentence_s2", repetitions=1)
    job = load_job(root)
    labels = job["task_specification"]["normalized_transcript"].split()
    grid = root / "mfa_output" / "record_000001.TextGrid"
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"',
             'item [1]:', '    name = "words"']
    for index, label in enumerate(labels, 1):
        lines.extend([f'    intervals [{index}]:', f'        xmin = {index / 10}',
                      f'        xmax = {index / 10 + .09}', f'        text = "{label}"'])
    lines.extend(['item [2]:', '    name = "phones"'])
    for index in range(1, len(labels) + 1):
        lines.extend([f'    intervals [{index}]:', f'        xmin = {index / 10}',
                      f'        xmax = {index / 10 + .09}', '        text = "B"'])
    grid.write_text('\n'.join(lines), encoding='utf-8')
    job['recordings'][0]['textgrid_path'] = str(grid)
    import json

    (root / 'job_manifest.json').write_text(json.dumps(job), encoding='utf-8')
    assert qc_job(root)['auto_pass'] == 1
    row = load_job(root)['recordings'][0]
    payload = load_recording_review(root, row)
    panel = WorkbenchReview()
    try:
        panel.set_record(root, row, payload)
        assert "Buy Bobby" in panel.canonical_transcript.toPlainText()
        assert panel.spoken_transcript.toPlainText() == job["task_specification"][
            "alignment_transcript"]
        assert panel.trial_selector.count() == 1
        panel._play_trial()
        assert panel._play_end_ms == 490
        panel.player.pause()
        panel.reviewer.setText('test_reviewer')
        panel._selected_word = 1
        panel.bound_start.setValue(.1)
        panel.bound_end.setValue(.2)
        panel._save_boundary()
        assert load_corrections(root, row['record_key'])['word_bounds']['1'] == [.1, .2]
        panel._render()
        assert len(panel._drag_lines) == 2
        panel._drag_lines[1].setValue(.195)
        panel._drag_boundary_finished("end", panel._drag_lines[1])
        assert load_corrections(root, row['record_key'])['word_bounds']['1'] == [.1, .195]
        requested = []
        panel.realign_requested.connect(lambda *args: requested.append(args))
        panel.spoken_transcript.setPlainText("Buy Bobby a puppy extra")
        panel._request_realign()
        assert requested == [(row['record_key'], "Buy Bobby a puppy extra", "test_reviewer")]
        panel._accept()
        assert load_job(root)['recordings'][0]['state'] == 'REVIEWED'
        assert load_corrections(root, row['record_key'])['decision'] == 'ACCEPT'
        app.processEvents()
    finally:
        panel.close()
