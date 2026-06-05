# VSLP Acoustic GUI v0.36 — Reports & Outputs refinement

This update removes the separate QC Dashboard stage from the Acoustic GUI workflow.

## Rationale

The dedicated Quality Control stage now extracts and summarizes recording-quality features after Data Segmentation. The previous QC Dashboard duplicated part of that role and appeared too late in the workflow. In v0.36, final review is handled by Reports & Outputs.

## Workflow

Current acoustic workflow:

1. Setup
2. Metadata
3. Preprocess
4. Data Segmentation
5. Quality Control
6. Feature Extraction
7. Inspector
8. Reports & Outputs

## Reports & Outputs

The Reports & Outputs tab now provides:

- stage reports
- primary tables
- run-level output manifest
- compact HTML run summary
- output-folder access

The generated run summary is an index/review artifact. It does not compute new QC decisions.

## New outputs

```text
acoustic/007_run_summary/tables/vslp_acoustic_output_manifest.csv
acoustic/007_run_summary/reports/vslp_acoustic_run_summary.html
acoustic/007_run_summary/manifests/stage_manifest.json
```
