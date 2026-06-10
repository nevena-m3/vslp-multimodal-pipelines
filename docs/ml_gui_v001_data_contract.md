# ML GUI v0.1 — Data Sources and Multimodal Dataset Contract

This patch starts the ML Modeling GUI with the correct foundation: data assembly and validation before model training.

## Why this stage exists

The ML GUI must not train models until it can verify that acoustic, kinematic, and metadata rows are aligned and leakage-safe. This stage builds unimodal and early-fusion datasets while preserving subject grouping and feature provenance.

## New acoustic exports

Acoustic aggregation now writes ML-ready outputs parallel to the kinematics exports:

- `acoustic/005_aggregation/tables/acoustic_features_ml_ready.csv`
- `acoustic/005_aggregation/tables/acoustic_features_only.csv`
- `acoustic/005_aggregation/tables/acoustic_features_canonical.csv`
- `acoustic/005_aggregation/tables/acoustic_feature_manifest.csv`
- `acoustic/005_aggregation/tables/acoustic_feature_manifest.json`

The full acoustic aggregation outputs remain unchanged.

## New ML dataset-contract outputs

The ML dataset builder writes:

- `ml/001_dataset/tables/ml_dataset_acoustic_only.csv`
- `ml/001_dataset/tables/ml_dataset_kinematic_only.csv`
- `ml/001_dataset/tables/ml_dataset_early_fusion.csv`
- `ml/001_dataset/tables/ml_feature_manifest.csv`
- `ml/001_dataset/tables/ml_modality_overlap.csv`
- `ml/001_dataset/tables/ml_row_exclusions.csv`
- `ml/001_dataset/tables/ml_dataset_manifest.json`

## GUI scope

The ML GUI currently supports:

- loading acoustic and kinematic ML-ready feature tables;
- loading metadata/label tables;
- loading acoustic and kinematic feature manifests;
- building acoustic-only, kinematic-only, and early-fusion datasets;
- computing modality overlap;
- exporting a combined ML feature manifest;
- running basic leakage prechecks.

It intentionally does not train models yet. Model training comes after feature-space selection and split/validation are implemented.

## Run command

```powershell
python -m vslp.gui.ml_app.main
```

## Test command

```powershell
python -m pytest tests/unit -k "acoustic_ml or ml_dataset" -q
python -m pytest tests/unit -q
```
