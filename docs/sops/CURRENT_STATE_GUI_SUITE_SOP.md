# VSLP GUI Suite: Current-State SOP and Menu Inventory

**Acoustic section updated:** `september-review`, commit `c8ce1fa` (17 September 2026). Other GUI sections remain an earlier inventory; verify them against current code before use.

## 1. What is in the repository

| Area | Current role | Entry point |
|---|---|---|
| Acoustic Pipeline GUI | Audio discovery, conversion, segmentation, QC, feature extraction, handoff | `vslp gui acoustic` or `vslp-acoustic-gui` |
| Kinematics Pipeline GUI | Video discovery, landmarks, selection, normalization, QC, features, aggregation, handoff | `vslp gui kinematics` or `vslp-kinematics-gui` |
| Feature Analysis GUI | One-modality feature audit, task review, recommendations, ML-ready export | `vslp gui features` or `vslp-features-gui` |
| ML Modeling GUI v0.1 | Prototype dataset-contract builder; no training or validation workflow | `python -m vslp.gui.ml_app.main` |
| CLI and backend | Shared project registry, stage runners, feature plugins, reports, data contracts | `src/vslp/cli`, `core`, `acoustic`, `analysis`, `ml` |
| Documentation and tests | Existing per-GUI SOPs, design/reference/history notes, unit and integration tests | `docs`, `tests` |

The three production-facing GUIs are PySide6 desktop applications. The ML GUI is present but is not exposed through the main `vslp gui` command. Feature Analysis runs separately for acoustic and kinematic inputs. Its optional Advanced ML Export Builder can align completed modality exports if the key is safe.

## 2. Start and workspace rules

1. Use Python 3.11 in an environment installed with `pip install -e ".[gui,silero,kinematic,dev]"`. Acoustic processing also needs FFmpeg/FFprobe; kinematic landmark extraction needs MediaPipe and a compatible `.task` model.
2. Keep raw audio/video and identifiable metadata in approved source storage. Choose a parent output folder outside the raw media. Acoustic Setup creates a task-stamped run folder beneath it.
3. Use the acoustic run folder as the handoff workspace for its downstream Feature Analysis run. Acoustic Setup initially creates `configs/`, `logs/`, and `acoustic/` beneath that run folder; the root `project_manifest.json` registers the project. Other components are not created by this Setup action.
4. Define the Setup task and acoustic recording policy before a production run. Study and clinical metadata are handled downstream in Feature Analysis. Pilot on representative files and inspect outputs before batch use.
5. Run a stage, inspect its table/report/error/manifest, and rerun downstream stages when upstream inputs or policies change. A green/completed indicator means the stage returned; it does not certify data validity.

### Handoff contract

Each upstream pipeline writes `acoustic/feature_handoff/main` or `kinematics/feature_handoff/main`. Four files are required: `feature_values.csv`, `feature_registry.csv`, `feature_status.csv`, and `feature_export_manifest.json`. `qc_features.csv` is optional for acoustic handoff. Kinematic handoff may also include `metadata_context.csv`. `feature_handoff/supplementary/artifact_catalog.csv` indexes detailed audit outputs. In Feature Analysis, **Load Main Handoff Folder** is the routine import path.

## 3. Acoustic Pipeline GUI: every main tab

**Launch:** `vslp gui acoustic`. The actual GUI is `src/vslp/gui/acoustic_app/main_window.py`. Its left rail shows Setup, Ingest, Preprocess, Segmentation, Segmentation manual review, Quality Control, Physiological Features, and Reports status. The bottom Run Log records stage messages. **Run Full Acoustic Workflow** runs Ingest once, then Preprocess and Segmentation, and pauses for manual review. After review decisions are frozen, **Continue to QC & Features** runs the remaining stages.

### Setup

- **Browse Input Folder / Browse Output Folder:** select source media and a parent output folder. Enter required project and task names; set recursive discovery as appropriate.
- **Initialize Project:** requires project name, task name, input folder, and output parent; creates a unique `TaskSlug_YYYYMMDD_HHMMSS` folder containing `project_manifest.json`, `configs/setup_config.json`, `logs/setup.log`, and `acoustic/`. Setup fields then lock and the generated run folder is displayed. Empty stage directories are removed after each stage. Clinical metadata is loaded in Feature Analysis.
- **Open existing run:** select the generated run folder to resume saved manual review after restarting the GUI. The manifest restores the locked Setup fields.
- **Run Ingest:** probes discovered media and records technical metadata, unsupported/read failures, duplicates, formats and codecs. The Setup display reports discovered, accepted, duplicate-skipped, and failed counts. Check counts against the source folder.

### Preprocess

- Configures canonical native-rate FLOAT32 audio creation. Optional processing is retained only where the GUI and stage explicitly support it; Silero's 16 kHz copy exists in memory during segmentation.
- **Run Preprocess:** creates canonical WAVs without changing originals. Review sample rates, duration, SNR, clipping, DC offset, failures and QC flags.

### Segmentation

- Choose **Silero VAD** for connected speech, **DDK energy-envelope** for repetitive oral DDK, or **Sustained phonation** for sustained vowels. The Setup task preselects a method where recognized; the user can change it. Only the selected method's parameters are shown. Custom remains a plugin slot.
- **Run Segmentation:** reads canonical audio and writes automatic intervals, frames, boundaries, a per-recording diagnostic plot, summary, review queue and stage manifest under `acoustic/002_segmentation/`. Inspect flags and boundaries; automatic outputs remain unchanged by later review.

### Segmentation manual review

- The queue opens on recordings awaiting review. Filters show all recordings, review-required recordings, automatic acceptances, kept automatic decisions, manual decisions, and exclusions. Select a recording to see status, flags, method, plot and audio playback.
- Enter reviewer name and notes, then **Keep automatic**, **Edit boundaries** and **Preview manual**, or **Exclude**. Manual intervals use one `start_sec,end_sec` pair per line and must fit within the recording without overlap. Decisions persist under `acoustic/003_segmentation_review/`.
- **Freeze final segmentation** requires decisions for all review-required recordings. It writes final decisions and intervals with boundary source and reviewer provenance. An automatic `EXCLUDED` can be recovered manually; a computational `FAILED` cannot be treated as a reviewed exclusion. Once frozen, **Continue to QC & Features** is available. To revise frozen boundaries, create a new run.

### Quality Control

- **Configure** subtab: choose QC families/features in a tree, inspect feature descriptions, select all/recommended/clear, set internal-pause, high-level percentile, hard-clip and near-clip thresholds, **Reset defaults**, then **Run Quality Control**. It requires frozen reviewed segmentation.
- Current families cover additive interference, gain dynamics, reverberation/echo, channel/device effects, nonlinear distortion and temporal discontinuity. The stage produces feature values, family status/scores, warnings and recommendations; warnings are review evidence.
- **Outputs** subtab: inspect summary tables, family/feature results, distributions, correlations, PCA/review ranking where available, plots, and **Open QC HTML report**. Missing output means it has not been generated or refreshed.

### Features

- Select feature families or individual features in the registry tree. Quick selectors cover all, implemented, implemented plus proxy, and clear. The detail panel describes scientific meaning, status, scale, support and computation notes; the policy table shows native scale and scalar-reduction decisions.
- Choose computation mode and acoustic region policy (`speech_only`, `effective_task`, or `full_file`), and minimum pause support. **Run Feature Extraction** requires frozen reviewed segmentation. Available plugin families include timing/respiratory, rhythm, phonatory, articulatory/formants, resonatory/nasality and coordination. An item marked proxy or pending should not be assumed to be a validated measure.
- Inspect per-file features, long status table, selected registry, expected-range flags, distribution audit, reduction audit and report before handoff.

### Inspector

- **Refresh latest outputs** finds stage artifacts; select a table/plot to preview it in **Table Preview** or **Plot Preview**. **Open current preview file** and **Open run folder** open the selected artifact/location.
- Reconcile input, segmented and feature row counts, file names, task context, errors and timestamps. A preview is a sample, not a complete audit.

### Reports & Outputs

- **Generate Run Summary:** writes an HTML stage summary and artifact manifest under `acoustic/006_run_summary`.
- **Open Main Feature GUI Handoff:** opens the canonical downstream folder. **Open Supplementary Outputs:** opens the audit catalog. **Open Acoustic Output Folder** and **Open Plots Folder** provide direct navigation.
- Confirm all four canonical acoustic handoff files, optional acoustic QC, and supplementary index before loading Feature Analysis. Join clinical metadata there.

## 4. Kinematics Pipeline GUI: every main tab

**Launch:** `vslp gui kinematics`. The active implementation is `src/vslp/gui/kinematics/app.py`. Its left rail reports stage status and offers **Refresh Latest Outputs**, **Run Full Kinematics Workflow**, and **Open Run Folder**. Check the Run Log and file-level errors after each stage.

### Setup

- Choose input video folder and shared workspace; set project/task settings. **Initialize Project** registers the component. **Run Video Ingest** inventories video format, FPS, dimensions, duration and readability. Verify discovery and technical warnings.

### Metadata

- **Browse Metadata** and **Run Metadata Linking** associate subject, task and visit context with ingested videos. Review unmatched and duplicate links.

### Landmarks

- Select a Face Landmarker `.task` model. **Download Default Model** and **Install / Verify MediaPipe Runtime** help set up the runtime; **Browse Model** chooses an existing file.
- Configure extraction parameters. **Write Landmark Extraction Plan** records intended settings; **Run MediaPipe Landmark Extraction** creates per-frame landmark evidence and detection coverage. Missing detections remain auditable.

### Landmark Selection

- **Refresh extracted videos**, choose a video/frame, **Load Frame** or **Load middle detected frame**, and inspect the real-frame overlay. Zoom In/Out/Face/Reset control the display.
- Select a preset or explicit landmarks, then **Apply / Save Selected Landmarks**. **Clear Selection** resets choices; **Save Overlay Preview PNG** saves visual evidence.
- Subtabs: **Selected landmarks** (current set), **Requirement checks** (required anchors/signals), **Preset reference** (preset contents), **Notes** (selection guidance). Confirm the selected landmarks are visible on real frames.

### Normalization

- Choose a normalization method and anchors. **Write Config** records settings; **Run Normalization** computes normalized trajectories; **Refresh Results** reloads tables.
- Subtabs: **Current method audit**, **Method comparison**, **Why normalize**. Review anchor validity, fallback use and invalid scales. Normalization does not correct pose, occlusion or tracking error.

### Video QC

- Configure implemented landmark/video checks; **Run Landmark / Video QC** and **Refresh QC Table** produce/reload detection, gap, anchor and normalization readiness evidence.
- Subtabs: **Current automated QC** (implemented results), **Future degradation families** (design catalog, not implemented validation), **Threshold provenance** (rationale/status). **Write QC Framework Placeholder** writes a planning artifact; it does not execute those future checks.

### Features

- Choose feature definitions with **Select all 65**, **Default oral-motor**, **Tier B only**, or **Clear**, and review requirements. **Write Feature Computation Plan** and **Write Feature Framework Catalog** document selection; **Run Kinematic Feature Computation** computes outputs; **Refresh Feature Outputs** reloads previews.
- Subtabs: **Full wide export** (engineering/support/QC columns too), **Canonical 65 / ML-ready** (curated scalar layer), **Column manifest** (roles/provenance), **Feature framework** (definitions), **Implementation audit** (what was actually computed), **QC requirements for selected features**. Do not use every wide-export column as an ML predictor.

### Aggregation

- Choose temporal aggregation/profile options. **Run Temporal Aggregation**, **Refresh Outputs**, and **Write Aggregation Guide** act on timeseries outputs.
- Subtabs: **Aggregated output**, **Profiles**, **Statistics guide**, **Design notes**. Review valid-frame fractions, support flags and reduction policy; keep timeseries for audit.

### Inspector

- **Refresh Inspector / Inventory** updates **Stage status**, **Readiness checklist**, **Artifact inventory**, **Latest table preview**, and **Inspector notes**. **Open Run Folder** opens the workspace. Verify timestamps and rerun downstream stages after changing selection or normalization.

### Reports & Outputs

- **Open Main Feature GUI Handoff** and **Open Supplementary Outputs** lead to canonical and audit folders.
- **Create Workflow Outline Report**, **Create Pipeline Summary Report**, **Open Latest HTML Report**, and **Open Run Folder** manage reporting.
- Subtabs: **Report preview**, **Report checklist**, **Report manifest**, **Output guidance**. Verify canonical files and the supplementary artifact catalog.

## 5. Feature Analysis GUI: every sidebar menu

**Launch:** `vslp gui features`. It analyzes **one modality at a time**. Its left sidebar has 15 pages; the Run Log stays visible. Most review pages populate only after accepted mappings and **Run Feature Analysis**. Common plot actions **Show**, **Regenerate**, and **Open current plot** display an existing plot, recompute a plot, or open its file. Plot and table selectors are exploratory views, not separate computation stages.

1. **Project:** Prefer **Load Main Handoff Folder**. It validates and populates primary features, status/registry and optional QC/metadata paths and infers modality/workspace. Direct table import uses individual browse fields, modality and output folder. **Load and Map Tables** loads inputs; **Run Feature Analysis** builds derived audit outputs after mapping.
2. **Feature Mapping:** review proposed roles; **Refresh Proposed Mapping** recomputes suggestions; **Accept Mapping and Continue** locks the working map. Roles include Feature, Identifier, Target, Covariate, QC and Ignore. Never map IDs, targets or QC metrics as predictors.
3. **Metadata Mapping:** assign subject, task, visit/session, outcomes and other context. **Refresh** and **Accept Metadata Mapping** establish joins. If metadata is absent, inspect a filename example and **Apply filename context** only with a documented naming rule. The GUI offers to run analysis after context acceptance.
4. **Overview:** design context, role mapping, feature families, feature quality and subject-by-task tables. Confirm row unit, subjects, tasks, mapping and family coverage.
5. **Missingness:** by-feature, by-row/recording, by-group, family and co-missing-pair views. Inspect support within each task and whether missingness clusters by subject or acquisition group.
6. **Distributions:** feature diagnostics, review summary, row outliers, expected ranges, shape audit and recording burden. Flags identify review candidates; they do not delete values.
7. **QC Integration:** summary, source variables, QC metric catalog, family burden, feature-by-QC and feature-by-family associations, missingness/outlier links, row burden. QC association is descriptive, not causal.
8. **Feature Relationships:** relationship summaries, correlation/redundancy and exploratory dimension structure/plots. Confirm that apparent redundancy is scientifically and task appropriate before reducing features.
9. **Task Review:** readiness summary, subject coverage, clinical balance, completeness, context detection and severity presets. Confirm task labels and denominator before interpreting any statistic.
10. **Longitudinal / Iterations:** repeated subject-task units, visit/session, iteration, date and feature-change audits, plus manual and automated QC change. Interpret change only when repeated-unit linkage is valid.
11. **Group / Outcome Screening:** priority effects, summary, variable catalog, continuous outcomes, categorical groups and balance. These are exploratory descriptive screens with sample-size and multiple-comparison limits.
12. **Reliability:** scoped repeatability outputs for supported same-task repeated observations. Review sample/support fields and source-task QC; missing support is not a favorable reliability estimate.
13. **Recommendations:** select a specific task; review readiness categories, priority queue, reasons and source evidence. Categories range from recommended through caution/review to exclude/recompute. A recommendation is an audit decision aid, not a model score.
14. **Export / Report:** select a specific task and profile (recommended plus caution, recommended only, review set, or full audit). **Refresh Preview** shows counts and manifest; **Create Export Package** writes matrix, target/covariate/QC tables, decision manifest, configuration, README, HTML report and ZIP. **Open Current Plot**, **Open HTML Report**, **Open Latest Package** inspect it. Use one package per intended task; All Tasks is an audit view.
15. **Advanced ML Export Builder:** optional import of completed acoustic and kinematic task exports; **Build ML Export Package** creates modality-specific outputs and early fusion only if a unique shared key is safe. Inspect **Export summary**, **Row alignment**, **Unified manifest**, **Acoustic ML-ready**, **Kinematic ML-ready**, **Early fusion**, **Exclusions / notes**, and **Package files**. **Open ML Export Folder** navigates to results. A failed alignment intentionally skips early fusion.

## 6. ML Modeling GUI v0.1: prototype inventory

Launch only with `python -m vslp.gui.ml_app.main` in the configured environment. This GUI is a data-contract workstation, **not a training application**.

- **Data Sources:** browse acoustic and kinematic ML-ready CSVs, metadata/labels CSV and both feature manifests. **Inspect Sources** summarizes rows/features; **Build ML Dataset Contract** writes aligned datasets; **Open ML Output Folder** opens results. Source-status cards show rows, features and matches.
- **Dataset Builder:** enter target column and optional comma-separated key columns; choose QC policy (`pass_review`, `pass_only`, `include_all`) and comparison mode (`maximum_available`, `matched_subjects`). Output subtabs are **Modality overlap**, **Acoustic-only dataset**, **Kinematic-only dataset**, **Early-fusion dataset**, **ML feature manifest**, and **Dataset manifest JSON**.
- **Modality & Feature Space:** read-only future-control description.
- **Splits & Validation:** read-only future-control description; subject-grouped validation is not implemented here.
- **Training / Evaluation / Registry:** read-only planned-work description; no model training, metrics or registry workflow.

## 7. Review checklist and current limits

- Verify source discovery, row counts, linkage, failures, QC and manifests at every transition. Preserve raw media and config provenance.
- The acoustic full-workflow action currently collects some widget values from its background worker. Treat this as a code-review item before relying on unattended batch execution.
- The three active GUIs are large desktop modules. This first pass checked launch routing and menu structure against source and existing SOPs. It did not run the GUIs or scientific pipelines on study media; this machine currently has no installed Python 3.11 environment in the clone.
- The ML GUI is a prototype; its placeholder pages must not be presented as working validation or modelling.
- Existing detailed SOPs under `docs/sops/` and data-contract references under `docs/reference/` remain the source for stage-specific parameters and output filenames. Historical notes under `docs/history/` describe earlier versions and should not override current code.

## 8. Cleanup performed in this review clone

- Removed three unused GUI scaffold implementations and their two package markers: `acoustic_app/main.py`, `kinematic_app/main.py` and `feature_analysis_app/main.py` plus the latter two `__init__.py` files. Active launchers and applications remain in `acoustic_app/app.py`, `kinematics/app.py` and `features/app.py`.
- Added the missing `__main__` launcher to `acoustic_app/app.py`, so the README's `python -m vslp.gui.acoustic_app.app` command now calls the actual window launcher.
- Added this current-state SOP. All changes are local and uncommitted for review.
