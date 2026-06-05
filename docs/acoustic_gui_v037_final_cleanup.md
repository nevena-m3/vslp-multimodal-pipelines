# VSLP Acoustic GUI v0.37 - Final GUI cleanup

This update adds institutional logos, a primary task field in Setup, a metadata-free full-run confirmation, and final workflow cleanup.

## Workflow

Full workflow now runs: Metadata, Ingest, Preprocess, Data Segmentation, Quality Control, Feature Extraction, and Run Summary.

If no metadata CSV is selected, the GUI asks the user to confirm before proceeding. The run can proceed without metadata; filename parsing and the Setup task fallback are used when available, while clinical labels remain blank unless provided.

## Setup task

The Setup tab now includes a task field. This is used as a fallback when metadata or filename parsing does not provide a task value.
