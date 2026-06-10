# Kinematics GUI v0.87 — Feature-only and ML-ready exports

## Purpose

The full kinematic feature output (`kinematic_features.csv`) is intentionally a wide research export. It includes metadata, QC/provenance columns, support-signal summaries, dense engineering summaries, and the canonical 65 scalar feature layer.

v0.87 keeps that full export unchanged and adds cleaner derivative tables for feature browsing and future ML workflows.

## New outputs

Written under:

```text
kinematics/006_features/tables/
```

| File | Purpose |
|---|---|
| `kinematic_features.csv` | Full wide research export. Preserved for audit and exploration. |
| `kinematic_features_canonical65.csv` | Row context plus canonical 65 scalar features. Best for the Feature GUI. |
| `kinematic_features_only.csv` | Only canonical numeric feature columns, no identifiers/QC/provenance. Useful for matrix-only ML input after row alignment is handled elsewhere. |
| `kinematic_features_ml_ready.csv` | `video_id` / `task_guess` plus canonical predictors. Best default input contract for the future ML GUI. |
| `kinematic_feature_manifest.csv` | Column manifest labeling each output column as canonical feature, metadata, QC metric, support signal, dense engineering summary, or other. |
| `kinematic_feature_manifest.json` | Machine-readable export summary and canonical feature list. |

## Column roles

The manifest uses these practical roles:

| Role | Meaning | Default ML use |
|---|---|---|
| `canonical_feature` | Curated 65 scalar kinematic feature layer. | Candidate predictor. |
| `metadata` | Video/source identity and file lineage. | Do not model directly. |
| `qc_metric` | QC/readiness/provenance metric. | Use for filtering/stratification, not disease prediction. |
| `support_signal` | Intermediate time-series signal used to derive scalar features. | Do not include by default. |
| `dense_engineering_summary` | Additional wide-export summary statistic. | Optional; review before modeling. |
| `other` | Uncategorized output column. | Review manually. |

## Design decision

The 339-column style table is still valuable as an engineering/research export. It should not be treated as a final ML design matrix by default. The ML GUI should default to `kinematic_features_ml_ready.csv` or `kinematic_features_canonical65.csv`, and should use `kinematic_feature_manifest.csv` to explain/organize columns.
