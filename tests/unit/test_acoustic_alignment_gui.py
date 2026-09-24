"""The optional Alignment UI is separate from feature execution."""

from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication

from vslp.gui.acoustic_app.alignment_widget import AlignmentWidget
from vslp.acoustic.alignment.self_test import MfaSelfTestResult


def test_alignment_task_first_controls_and_advanced_import(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    widget = AlignmentWidget(lambda: tmp_path)
    emitted = []
    widget.run_requested.connect(emitted.append)
    assert widget.words._alignment_row.isVisible() is False
    widget.show()
    assert not widget.words._alignment_row.isVisible()
    assert not widget.prompt._alignment_row.isVisible()
    assert widget.source.currentData() == "mfa"
    assert widget.run_button.text() == "RUN ALIGNMENT"
    assert not widget.run_button.isEnabled()
    widget.advanced_toggle.click()
    widget.source.setCurrentIndex(1)
    assert widget.words._alignment_row.isVisible()
    assert widget.model.isVisible()
    widget.prompt.setText(str(tmp_path / "prompt.json"))
    widget.words.setText(str(tmp_path / "words.csv"))
    widget._run()
    assert emitted[-1].source == "external"
    assert emitted[-1].words_csv.endswith("words.csv")
    assert widget.profile._alignment_row.isVisible()
    assert widget.model.isVisible()
    assert "environment" in widget.environment.text().lower()
    assert widget.self_test_button.text() == "RUN MFA SELF-TEST"
    widget.profile.setText(str(tmp_path / "missing_profile.json"))
    requests = []
    widget.environment_requested.connect(requests.append)
    widget._check_environment()
    assert requests[-1].endswith("missing_profile.json")
    assert "Checking" in widget.environment.text()
    self_tests = []
    widget.self_test_requested.connect(lambda profile, wav: self_tests.append((profile, wav)))
    widget.self_test_button.click()
    assert self_tests == [(str(tmp_path / "missing_profile.json"), "")]
    widget.show_self_test_result(MfaSelfTestResult(
        result="PASS", word_count=5, phone_count=5, elapsed_time_sec=1.5))
    assert "SELF-TEST PASSED" in widget.self_test_summary.text()
    widget.show_self_test_result(MfaSelfTestResult(
        result="FAIL", failure_step="MFA validation", failure_message="OOV"))
    assert "SELF-TEST FAILED" in widget.self_test_summary.text()
    assert "MFA validation" in widget.self_test_summary.text()
    widget.close()
    assert app is not None
