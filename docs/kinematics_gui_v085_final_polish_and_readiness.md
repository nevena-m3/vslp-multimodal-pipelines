# Kinematics GUI v0.85 — Final polish and readiness pass

This release is a conservative clean-up pass before starting the ML-models GUI.
It does not change landmark extraction, normalization mathematics, QC decisions,
feature formulas, or aggregation formulas.

## What changed

- Updated the GUI version label to `v0.85`.
- Updated the top-level GUI module description so it reflects the current full kinematics workflow rather than the early scaffold state.
- Cleaned kinematics documentation test commands for Windows PowerShell.
- Replaced fragile pytest file-glob examples with the reliable command:

```powershell
python -m pytest tests/unit -k kinematics -q
```

## Recommended validation commands on Windows

```powershell
cd C:\VSLP\vslp-multimodal-pipelines
.\.venv-win\Scripts\Activate.ps1
git switch win-dev

python -m pytest tests/unit -k kinematics -q
python -m pytest tests/integration/test_acoustic_preprocess_stage.py -q
python -m vslp.gui.kinematics.app
```

For a full repository check, run:

```powershell
python -m pytest -q
```

## Functional readiness checklist

Before moving to the ML-models GUI, run one end-to-end kinematics smoke test:

1. Setup / Project: initialize project and run video ingest.
2. Metadata: link metadata if available.
3. Landmarks: verify MediaPipe runtime and run landmark extraction.
4. Landmark Selection: load a real frame, inspect overlay, save selected landmarks.
5. Normalization: run normalization and confirm scale-source / QC diagnostics.
6. Video QC: run current automated QC and confirm the future degradation-family framework is clearly marked as to-be-built.
7. Features: select the 65-feature scalar layer and compute features.
8. Aggregation: run robust aggregation and confirm guide files are written.
9. Inspector: refresh inventory and verify generated artifacts.
10. Reports & Outputs: create workflow and pipeline summary reports.

## Scope note

The kinematics GUI is ready for integrated workflow testing. It is not yet a
validated clinical diagnostic system. QC thresholds, disease-specific model
thresholds, and visual degradation QC need dataset-level calibration and future
validation.
