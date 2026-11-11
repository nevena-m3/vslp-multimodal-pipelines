"""The Acoustic GUI imports alignment results; it never runs or loads an aligner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

# Modules that construct, configure, launch or run Montreal Forced Aligner, or
# that implement the alignment producer/review GUI.
FORBIDDEN = (
    "vslp.acoustic.alignment.mfa_provider",
    "vslp.acoustic.alignment.mfa_launcher",
    "vslp.acoustic.alignment.self_test",
    "vslp.acoustic.alignment.stage",
    "vslp.acoustic.alignment.task_workflow",
    "vslp.acoustic.alignment.workbench",
    "vslp.acoustic.alignment.cancellable_runner",
    "vslp.gui.alignment_app",
)

PROBE = """
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from vslp.gui.acoustic_app.main_window import AcousticPipelineWindow
window = AcousticPipelineWindow()
from vslp.acoustic.alignment import load_final_alignment, AlignmentStore
from vslp.acoustic.alignment.package_v1 import import_alignment_package
import vslp.acoustic.features.mixed_dispatch
import vslp.acoustic.features.family04_stage, vslp.acoustic.features.family06_v2_stage
import vslp.acoustic.features.family07_v2_stage, vslp.acoustic.features.family08_v2
print("\\n".join(sorted(sys.modules)))
"""


def test_acoustic_gui_and_feature_path_load_no_aligner_modules():
    result = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                            env={"PYTHONPATH": str(SRC), "QT_QPA_PLATFORM": "offscreen",
                                 "PATH": ""}, timeout=300, check=True)
    loaded = set(result.stdout.split())
    offenders = sorted(name for name in loaded
                       if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN))
    assert offenders == []


def test_acoustic_app_source_has_no_mfa_controls():
    source = (SRC / "vslp" / "gui" / "acoustic_app").glob("*.py")
    text = "\n".join(path.read_text(encoding="utf-8") for path in source)
    for token in ("run_mfa_self_test", "check_mfa_environment", "run_task_alignment",
                  "run_acoustic_alignment", "MfaProvider", "AlignmentWidget"):
        assert token not in text
    assert "Import Alignment Package" in text
