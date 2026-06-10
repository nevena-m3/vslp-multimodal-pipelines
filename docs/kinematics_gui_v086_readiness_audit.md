# Kinematics GUI v0.86 — readiness audit and final cleanup

This release adds a compact readiness checklist to the Inspector so the kinematics GUI can be assessed before moving into the ML-models GUI.

## What was cleaned

- Updated the GUI version to `v0.86`.
- Removed stale internal method names that still used `placeholder` for feature computation and aggregation, even though both stages now run real backend code.
- Added an Inspector **Readiness checklist** tab.
- Added readiness checklist exports:
  - `kinematics/008_inspector/tables/readiness_checklist.csv`
  - `kinematics/008_inspector/tables/readiness_checklist.json`
- Kept the visual-degradation QC framework clearly marked as deferred work, not a completed or validated QC protocol.

## What still is not perfect

The kinematics GUI is interface-ready, but it is not scientifically “finished” in the sense of validation/calibration. The known remaining work is:

1. **Visual-degradation QC framework** — lighting, freezing/dropped frames, camera instability, occlusion/cropping, multiple faces, pose, and task compliance are still a clearly marked future module.
2. **Threshold calibration** — QC thresholds are conservative engineering defaults and should be calibrated on the user’s actual VSLP datasets.
3. **Disease interpretation** — ALS/Parkinson’s interpretation should not be treated as diagnostic until validated against labeled cohorts.
4. **End-to-end Windows GUI smoke testing** — unit tests cover backend and source-level behavior, but the GUI still needs manual workflow testing with real videos on Windows.
5. **Model-readiness validation** — the ML GUI should consume source CSV/JSON artifacts, not report HTML files.

## Recommended Windows test command

Use the PowerShell-safe command:

```powershell
python -m pytest tests/unit -k kinematics -q
python -m vslp.gui.kinematics.app
```

Do not pass wildcard-style individual test-file paths to pytest in PowerShell; use the folder plus `-k kinematics` pattern above instead.
