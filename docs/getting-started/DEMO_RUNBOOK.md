# Acoustic, Kinematics, and Feature GUI Demo Runbook

## Scope

This runbook demonstrates the completed upstream workflow only. It does not demonstrate model training or clinical diagnosis.

## Preflight

1. Use Python 3.11 and install `.[gui,kinematic,dev]`. Install `.[silero]` separately when the demo requires Silero VAD.
2. Verify `ffmpeg -version`, `ffprobe -version`, and `python -m vslp.cli.main doctor`.
3. Prepare a small de-identified audio/video subset with known task and recording linkage.
4. Create one empty, writable study workspace outside the source-media folders.
5. Confirm the MediaPipe Face Landmarker model before the Kinematic demo.

## Acoustic Demo

1. Launch `python -m vslp.cli.main gui acoustic`.
2. Select audio input and the shared study workspace.
3. Run metadata, preprocessing, segmentation, QC, and feature extraction using the approved demo configuration.
4. Review file counts, warnings, segmentation examples, QC summaries, feature status, and formula registry.
5. Open **Reports & Outputs**.
6. Open **Main Feature GUI Handoff** and verify the four canonical files.
7. Open **Supplementary Outputs** and show `artifact_catalog.csv` as the audit index.

## Kinematics Demo

1. Launch `python -m vslp.cli.main gui kinematics`.
2. Select video input and the same study workspace.
3. Verify MediaPipe, select the approved landmark preset, and inspect a real-frame overlay.
4. Run the ordered workflow or each stage in sequence.
5. Review detection coverage, normalization anchors/fallbacks, QC warnings, canonical features, and temporal aggregation support.
6. Open **Reports & Outputs**.
7. Open **Main Feature GUI Handoff** and verify the four canonical files.
8. Open **Supplementary Outputs** and show the artifact catalog, timeseries, and visual/QC evidence.

## Feature Analysis Demo

1. Launch `python -m vslp.cli.main gui features`.
2. Select **Load Main Handoff Folder** and choose one modality's `feature_handoff/main` folder.
3. Confirm that feature and registry paths, modality, and workspace are filled automatically. Confirm whether optional QC and metadata were present.
4. Review and accept Feature Mapping and Metadata Mapping.
5. Run Feature Analysis once when prompted.
6. Demonstrate task-specific Missingness, QC, Relationships, Screening, Reliability, and Recommendations.
7. In Export / Report, select one task and create the task-specific package.
8. Verify its feature matrix, target/covariate/QC tables, decision manifest, configuration, README, HTML report, and ZIP.

## Acceptance Checklist

- All three GUIs launch without traceback.
- Raw media remains unchanged.
- Acoustic and Kinematic main handoffs share the same filenames and registry schema.
- Missing optional metadata or QC is reported, never silently invented.
- Stage warnings and unsupported features remain visible.
- Feature Analysis loads a whole main handoff in one action.
- Recommendations and exports are task-specific.
- No identifiers, paths, QC metrics, imputation, scaling, or feature selection are presented as default predictors.
- Generated outputs and participant data are not committed to Git.

## Verified Release Checks

Run from the repository root:

```powershell
python -m pytest -q --basetemp=.pytest_tmp tests/unit
python -m ruff check src tests
python -m py_compile src/vslp/gui/acoustic_app/main_window.py src/vslp/gui/kinematics/app.py src/vslp/gui/features/app.py
```

Use a repository-local `--basetemp` if institutional Windows permissions block the default AppData temp directory.
