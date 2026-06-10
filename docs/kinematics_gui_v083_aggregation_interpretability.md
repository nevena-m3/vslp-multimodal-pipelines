# VSLP kinematics GUI v0.83 — Aggregation interpretability

## Purpose

This patch refines the Aggregation menu so the analyst can understand exactly what is being collapsed from time series into per-video scalar outputs.

Aggregation is a compression policy. It does not replace the frame-level feature time series. The frame-level CSVs remain the audit layer, while the aggregation stage writes a separate scalar table for modeling, export, and report-level summaries.

## Key design principle

A kinematic signal can contain one value per frame, for example mouth aperture or lip-spread velocity. A single video therefore has a time series with N frame values. Aggregation summarizes those values into robust scalar descriptors such as median, IQR, p05, p95, and p95-p05 range.

Mean is not the default scientific answer because it can hide peaks, bursts, pauses, tracking spikes, and missing regions. Robust statistics are safer for markerless video.

## GUI changes

- Adds Aggregation status cards.
- Reorganizes the Aggregation tab into a larger policy/settings area with output tables at the bottom.
- Adds a built-in explanation of the collapse from frame-level values to per-video scalar summaries.
- Adds a Statistics Guide tab explaining median, IQR, p05/p95, valid fraction, profiles, and settings.
- Adds a Write Aggregation Guide button.

## Backend changes

New aggregation guide outputs:

- `kinematics/007_aggregation/tables/aggregation_guide.csv`
- `kinematics/007_aggregation/tables/aggregation_guide.json`

The guide is also written automatically when temporal aggregation is run.

## Validation

Validated with:

```powershell
PYTHONPATH=src python -m pytest tests/unit/test_kinematics_*.py -q
PYTHONPATH=src python -m pytest tests/integration/test_acoustic_preprocess_stage.py -q
python -m compileall -q src/vslp/analysis/kinematics/aggregation.py src/vslp/gui/kinematics/app.py
```

Expected results in the patch build:

- 57 kinematics unit tests passed
- 1 acoustic integration test passed
- compile checks passed
