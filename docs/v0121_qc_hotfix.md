# VSLP v0.12.1 QC Hotfix

This patch fixes a GUI stability issue observed when opening/running the QC Dashboard from the PySide6 desktop GUI.

## Root cause

Matplotlib plot generation was being called from a GUI worker thread while Matplotlib could select an interactive Qt backend. On macOS this can destabilize or crash a Qt application, especially when QC plots are generated from the GUI.

## Fix

All backend plot-generating stages now force Matplotlib's non-interactive `Agg` backend before importing `matplotlib.pyplot`:

- acoustic segmentation diagnostic plots
- acoustic feature plots
- acoustic aggregation plots
- acoustic QC dashboard plots

This keeps plotting headless and file-based. The GUI only loads finished PNG files after they are written to disk.

## Expected behavior

Running QC from the GUI should now write:

```text
acoustic/006_qc_dashboard/tables/acoustic_qc_dashboard.csv
acoustic/006_qc_dashboard/tables/qc_flag_counts.csv
acoustic/006_qc_dashboard/plots/qc_pass_review_counts.png
acoustic/006_qc_dashboard/plots/qc_snr_distribution.png
acoustic/006_qc_dashboard/plots/qc_speech_fraction_distribution.png
acoustic/006_qc_dashboard/plots/qc_flag_counts.png
acoustic/006_qc_dashboard/reports/acoustic_qc_dashboard.html
```

If QC fails for data reasons, the GUI should show an error dialog instead of crashing.
