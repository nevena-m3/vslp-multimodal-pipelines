# VSLP Feature Analysis GUI Standard Operating Procedure

## 1. Purpose

The Feature Analysis GUI audits one feature modality at a time, acoustic or kinematic, before machine-learning development. It helps clinicians and researchers inspect data structure, missingness, distributions, QC sensitivity, feature relationships, task coverage, longitudinal support, group/outcome screening, reliability, and task-specific readiness recommendations.

The GUI does not train models. Its canonical handoff is the task-specific package created in **Export / Report**. Imputation, scaling, transformation, feature selection, data splitting, model fitting, and validation belong in the downstream ML GUI and must occur inside training or cross-validation folds.

## 2. Scope and Responsibilities

- Use one modality per Feature Analysis run. Acoustic and kinematic runs may share a study workspace because their outputs are isolated by modality.
- Confirm that rows represent the intended recording or analysis unit.
- Review all automatically proposed column roles before analysis.
- Use accepted metadata as the primary source of subject, task, visit, outcome, and covariate context.
- Use filename-derived context only when a metadata table is unavailable.
- Treat recommendations and screening statistics as descriptive decision support, not confirmatory inference or automated feature selection.
- Export a separate ML handoff for each task unless the scientific protocol explicitly defines a valid pooled-task model.

## 3. Required Inputs

### Recommended modality handoff

Preferred. Select `acoustic/feature_handoff/main` or `kinematics/feature_handoff/main` with **Load Main Handoff Folder**. The GUI validates the contract, fills feature/registry/QC/metadata paths, infers modality, selects the shared workspace, and loads the tables. This is the demo and routine operating path.

### Primary feature table

Required. One row per recording or analysis unit, with numeric extracted features and enough identifiers to trace each row.

### Metadata table

Strongly recommended. It should contain stable join keys and relevant task, subject, visit/session, target/outcome, group, and covariate fields.

### QC table

Optional but recommended. It may contain manual QC flags, automated quality metrics, or both. QC variables are used to assess sensitivity of features to recording or extraction quality.

### Feature registry

Optional but recommended. It may define feature family, expected range, units, interpretation, or policy information.

## 4. Start the GUI

From the repository root:

```powershell
cd C:\VSLP\vslp-multimodal-pipelines
.\.venv-win\Scripts\pythonw.exe -m vslp.gui.features.app
```

For a visible terminal and diagnostic output, use `python.exe` instead of `pythonw.exe`.

## 5. Standard Workflow

### Step 1: Project

1. Prefer **Load Main Handoff Folder** and select the Acoustic or Kinematic `feature_handoff/main` folder.
2. Confirm the detected modality, workspace, optional QC, and optional metadata shown in the Run Log.
3. For legacy or independent tables, use Direct Table Import, select the primary table and optional QC/metadata/registry, then select modality and workspace manually.
4. Confirm the Run Log reports the expected row and column counts.

Stop if the feature table is empty, the row unit is unclear, or duplicate columns cannot be explained.

### Step 2: Feature Mapping

1. Review every proposed role.
2. Assign extracted measurements to **Feature**.
3. Assign stable row keys to **Identifier**.
4. Assign clinical or experimental outcomes to **Target**.
5. Assign adjustment variables to **Covariate**.
6. Assign recording or extraction quality variables to **QC**.
7. Ignore administrative or unsupported fields.
8. Accept the mapping and continue.

Never classify a target, diagnosis, subject identifier, task label, visit label, or QC measurement as a predictor feature.

### Step 3: Metadata Mapping

When metadata is loaded, verify the semantic roles and accept the mapping. When metadata is unavailable, configure the filename source and parsing template, inspect an example, and apply filename-derived context.

After context is accepted, the GUI asks once whether to run Feature Analysis. Select **Yes** for the standard workflow. If analysis is deferred, use **Run Feature Analysis** on the Project page later; other menus remain navigable but show passive empty states.

### Step 4: Overview and Data Quality Review

Review menus in this order:

1. **Overview**: confirm subjects, tasks, row counts, role counts, and feature-family coverage.
2. **Missingness**: inspect feature and row missingness within each task.
3. **Distributions**: review variance, outliers, expected-range flags, and implausible shapes.
4. **QC Integration**: identify features associated with manual or automated QC burden.
5. **Feature Relationships**: inspect redundancy, correlation structure, and exploratory PCA.
6. **Task Review**: verify task labels, subject coverage, completeness, and clinical balance.
7. **Longitudinal / Iterations**: confirm repeated-measure support and visit/session structure.
8. **Group / Outcome Screening**: review descriptive associations and sample support.
9. **Reliability**: evaluate same-task repeatability only where repeated observations support estimation.

Interpret task filters literally. A result from one task must not be generalized to another task without a scientific justification and supporting analysis.

### Step 5: Recommendations

1. Select a specific task.
2. Review the readiness counts and priority queue.
3. Inspect the reasons and recommended action for each feature.
4. Return to the named source menu when a recommendation flags missingness, distribution, QC, redundancy, screening, or reliability concerns.
5. Document any decision to retain a review or exclusion feature.

Readiness categories are review defaults:

- **Recommended**: clean default candidate.
- **Recommended with caution**: usable with a documented risk and sensitivity plan.
- **Review before use**: hold until an analyst resolves the stated concern.
- **Exclude or recompute**: correct the measurement or retain only with explicit justification.
- **Exclude by default**: insufficient support for the default ML handoff.

## 6. Export / Report

### Task focus

The first available specific task is selected by default. Create one package per task. **All tasks** is an audit view and should not be used as the default ML handoff.

### Export profile

- **Recommended + caution**: standard starting set, with caution reasons retained.
- **Recommended only**: strict sensitivity set.
- **Review set**: broader analyst-review or sensitivity set.
- **Full audit**: all features; not an ML default.

### Required review before export

Confirm that:

- the selected task is correct;
- included and held counts are plausible;
- the manifest plot is visible;
- identifiers and context columns are present;
- target, covariate, and QC roles are correct;
- no target or subject identifier is included as a predictor feature.

Select **Create Export Package**. Review the Package Contents / Status table and open the latest package.

### Package contents

- `ml_ready_feature_matrix.csv`
- `ml_target_table.csv`
- `ml_covariate_table.csv`
- `ml_qc_covariate_table_from_feature_table.csv`
- `feature_export_manifest.csv`
- `feature_recommendation_summary.csv`
- `export_profile_summary.csv`
- `feature_recommendation_legend.csv`
- `export_config.json`
- `README.md`
- `vslp_feature_analysis_export_report.html`
- ZIP archive of the package

The manifest and configuration record task scope, profile, row count, included features, role columns, provenance, and the leakage-safe modelling boundary.

## 7. Advanced ML Export Builder

Use this optional page only to standardize or align acoustic and kinematic outputs. It does not replace the task-specific Export / Report package.

Safe early fusion requires a shared unique key. Preferred keys are:

1. `record_key`
2. `subject_id + session_id + task`
3. `subject_id + visit_id + task`
4. `subject_id + task`

If alignment is unsafe, early fusion is intentionally skipped. Review the Row Alignment and Exclusions / Notes tables. Modality-specific outputs remain available.

## 8. Scientific Interpretation Rules

- Missingness percentages describe support; they do not justify a specific imputation method.
- Outlier flags identify observations for review; they are not automatic deletion rules.
- QC correlations indicate sensitivity or association, not causality.
- Redundancy is task-specific and does not prove that one feature is scientifically dispensable.
- Group/outcome screening is exploratory and may involve multiple comparisons, small groups, confounding, and repeated observations.
- Reliability estimates require adequate repeated observations of the same task and unit. Unsupported estimates must not be reported as stable.
- Recommendation scores summarize transparent rules; they are not predictive performance estimates.
- Feature selection and all learned preprocessing must occur within training folds in the ML GUI.

## 9. Troubleshooting

### Menus are empty

Confirm that mappings were accepted and Feature Analysis completed. Check the Run Log for the first error.

### No tasks are available

Verify Metadata Mapping or filename-derived task parsing. Confirm that the mapped task column contains non-empty values.

### Export plot is unavailable

Run Feature Analysis, select a task, and select **Refresh Preview**. Confirm the recommendation manifest is not empty.

### Early fusion is skipped

Review the Row Alignment table. Add or repair a safe shared recording key; do not force a many-to-many merge.

### Export fails

Confirm the output folder is writable, the selected task has rows, and mapped columns still exist in the active feature table. Preserve the Run Log error for investigation.

## 10. Completion Checklist

- Inputs and row unit verified.
- Feature and metadata mappings accepted.
- Analysis completed without unresolved errors.
- Task-specific quality, screening, and reliability reviewed.
- Recommendation reasons reviewed and exceptions documented.
- One task-specific Export / Report package created per intended ML task.
- Package tables, manifest, configuration, report, and ZIP verified.
- Advanced multimodal alignment reviewed if used.
- Handoff delivered to the ML workflow without pre-splitting leakage-prone transformations.
