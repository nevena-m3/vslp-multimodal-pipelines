# VSLP Kinematics Pipeline Standard Operating Procedure

## 1. Purpose

This SOP defines controlled use of the VSLP Kinematics Pipeline GUI to create auditable facial-landmark trajectories, normalized signals, QC evidence, and per-video kinematic feature exports for research.

The pipeline does not recover true millimeter biomechanics from ordinary video, validate task compliance, or provide diagnostic interpretation. Current automated extraction QC is implemented; the broader visual-degradation QC framework remains under development and is not a validated protocol.

## 2. Inputs and Preconditions

Required:

- source videos readable by OpenCV/MediaPipe;
- a writable shared VSLP study workspace outside the source-video directory;
- Python 3.11 environment with GUI and kinematic dependencies;
- compatible MediaPipe Face Landmarker `.task` model;
- defined task, row/video unit, and landmark policy.

Recommended:

- metadata CSV with subject, session/visit, task, and source-file linkage;
- mostly frontal face visibility and adequate image resolution;
- documented camera, framing, lighting, and acquisition protocol;
- pilot review across representative subjects, tasks, poses, and image quality.

## 3. Launch

```text
vslp gui kinematics
```

## 4. Controlled Workflow

### Step 1: Setup and Video Ingest

1. Select the source-video folder and the shared VSLP study workspace. Kinematics artifacts will be written under `kinematics/`.
2. Initialize the project.
3. Run Video Ingest.
4. Review format, FPS, duration, resolution, readability, and warnings.

Acceptance checks:

- expected videos are discovered once;
- identifiers are unique and traceable;
- unreadable or anomalous media are documented;
- frame-rate and resolution variability are understood.

### Step 2: Metadata

Load and link metadata where available. Review unmatched and duplicate rows. Do not infer participant or clinical fields from filenames unless the convention is explicit and validated.

### Step 3: Face Landmarks

1. Install or verify the MediaPipe runtime/model.
2. Review the extraction plan and parameters.
3. Run Face Landmark extraction.
4. Inspect the manifest and detection coverage.

Frames without a detected face are retained as missing landmark rows for auditability. They must not be silently removed before coverage is assessed.

### Step 4: Landmark Selection

1. Load a representative detected frame.
2. Inspect the real-frame landmark overlay.
3. Select a validated preset or explicit landmark set.
4. Confirm required normalization anchors.
5. Save the selection and requirement checks.

The oral-motor core preset is a starting point, not a universal validated set. Custom selections require a documented scientific rationale.

### Step 5: Normalization

Use the study-approved normalization method. Intercanthal scaling is the preferred default for oral/jaw kinematics when anchors are visible. Fallback anchors are recorded and QC-flagged.

Normalization reduces camera-distance and face-size effects. It does not correct head rotation, poor tracking, occlusion, camera motion, or perspective distortion.

Review normalized landmark coverage, anchor validity, fallback use, and invalid-scale rows.

### Step 6: Video and Landmark QC

Run the implemented QC stage and inspect detection coverage, missing runs/gaps, normalization readiness, anchor stability, and output warnings.

The future degradation-family catalog is a design placeholder. Lighting, blur, freezing/dropped frames, camera instability, occlusion/cropping, multiple faces, pose, and task-compliance thresholds require dataset-specific validation.

### Step 7: Feature Computation

1. Select canonical feature definitions.
2. Confirm required landmarks, native signals/units, and preprocessing settings.
3. Run feature computation.
4. Review the canonical feature table, wide export, manifest, timeseries, implementation audit, and feature-specific QC requirements.

Use the canonical feature layer as the default Feature Analysis input. The full wide table includes support, QC, provenance, and engineering summaries that must not all be treated as predictors.

### Step 8: Temporal Aggregation

Choose and document the scalar aggregation profile. Review valid-frame fractions and low-support flags. Aggregation compresses timeseries; it does not replace timeseries or visual review.

### Step 9: Inspector

Refresh the Inspector and verify stage status, readiness checklist, artifact inventory, timestamps, and latest-table preview. Rerun downstream stages after changing landmarks, normalization, or QC inputs.

### Step 10: Reports and Outputs

Generate the report package and verify the report checklist and manifest. Reports summarize evidence; they are not data inputs and do not replace QC review.

## 5. Main and Supplementary Outputs

In **Reports & Outputs**, use **Open Main Feature GUI Handoff** for downstream Feature Analysis. Use **Open Supplementary Outputs** for audit, diagnostics, timeseries, reports, and troubleshooting.

Main downstream folder:

```text
kinematics/feature_handoff/main/
|-- feature_values.csv
|-- feature_registry.csv
|-- feature_status.csv
|-- feature_export_manifest.json
|-- qc_features.csv              # when available
|-- metadata_context.csv         # when available
`-- README.md
```

The four canonical files are required. Metadata is recommended but extraction may proceed without it; absent metadata is recorded as `no_metadata_provided` and can be supplied in Feature Analysis.

Supplementary index:

```text
kinematics/feature_handoff/supplementary/artifact_catalog.csv
```

The established stage outputs remain in place:

```text
kinematics/000_ingest/
kinematics/001_metadata/
kinematics/002_landmarks/
kinematics/003_selection/
kinematics/004_normalization/
kinematics/005_video_qc/
kinematics/006_features/
kinematics/007_aggregation/
kinematics/008_inspector/
kinematics/009_reports/
```

Legacy and detailed feature-stage outputs include:

```text
kinematics/006_features/tables/kinematic_features_canonical65.csv
kinematics/006_features/tables/kinematic_features_ml_ready.csv
kinematics/006_features/tables/kinematic_feature_manifest.csv
kinematics/005_video_qc/tables/landmark_video_qc_summary.csv
```

Audit outputs include the full `kinematic_features.csv`, normalized landmark manifest, per-frame timeseries, aggregation outputs, readiness checklist, and report manifest. They remain supplementary because scalar feature handoff must not replace timeseries or visual QC review.

## 6. Stop Conditions

Stop and resolve the issue when:

- videos are unreadable or systematically inconsistent with the protocol;
- face detection or landmark coverage is inadequate;
- the selected landmarks are not visible on real frames;
- normalization anchors are missing or unstable for a substantial subset;
- QC indicates long missing runs or implausible tracking;
- feature rows cannot be traced to videos and subjects;
- canonical features are confused with metadata, QC, or engineering columns;
- output counts or timestamps cannot be reconciled.

## 7. Completion Checklist

- Source videos preserved unchanged.
- Ingest warnings reviewed.
- Metadata linkage reviewed.
- Runtime and model provenance recorded.
- Real-frame landmark overlays inspected.
- Selection and normalization policy documented.
- Landmark/video QC reviewed with limitations acknowledged.
- Canonical feature and timeseries outputs audited.
- Aggregation policy recorded if used.
- Inspector readiness checklist reviewed.
- Main handoff opened and its four canonical files verified.
- Optional QC and metadata context presence or absence documented.
- Supplementary artifact catalog retained with the study record.
