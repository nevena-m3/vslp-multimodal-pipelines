# VSLP Feature Analysis GUI v0.41 — Visual and Column Mapping Refinement

This update substantially revises the visual design of the Feature Analysis GUI and strengthens column-role detection.

## Visual changes

- White top branding bar with dark navy text and no dark highlight artifacts.
- Fixed-size logo containers for the Speech Production Lab and University of Toronto logos.
- Dark navy left sidebar aligned with the Acoustic GUI product family.
- Cleaner professional card layout, restrained borders, better spacing, and larger working area.
- Better modality dropdown styling and sizing.
- More readable table styling for mapping and output previews.

## Column mapping changes

The role detector is now intentionally conservative. In the primary feature table, numeric columns are treated as features unless a stronger rule identifies them as identifiers, QC variables, audit/status variables, covariates, or exact clinical labels.

This prevents acoustic/kinematic features from being misclassified as targets.

Target/label assignment now requires exact or near-exact clinical label names, such as diagnosis, severity_score, severity_bin, ALSFRS total, ALSBDI, target, label, or outcome. Generic words like score are not enough to make something a target.

Users can manually override roles in the Column Mapping page.

## Supported roles

- Identifier
- Feature
- QC feature
- Target / label
- Covariate
- Time / visit
- Audit / status
- Ignore

## Launch

```bash
python -m vslp.gui.features.app
```

or

```bash
python tools/run_feature_analysis_gui.py
```
