# Kinematics GUI v0.75 - Landmark Selection Workstation hardening

This patch refines the Landmark Selection workstation after user review of v0.74.

## GUI changes

- Replaces the `Prev`, `Next`, `-10`, and `+10` frame buttons with a scrubber-first workflow.
- Adds `Load Selected Frame` for exact frame loading.
- Adds `Use Representative Detected Frame`, which jumps to the middle frame where `face_detected=True` and loads the overlay.
- Keeps the exact frame numeric entry but hides spinbox arrow buttons.
- Renames the auto-load option to `Auto-load while scrubbing`.
- Adds overlay style control: `Subtle`, `Balanced`, and `High contrast`.
- Adds toggles for selected landmark ID labels and guide edges.
- Changes the default overlay from aggressive high-contrast green to a restrained review-grade cyan/gold design.

## Scientific correction

The previous default oral-motor preset contained outer eye anchors `33/263`, but the default normalization method expected inner-canthus anchors `133/362`. That mismatch could make the Selection dashboard report intercanthal-anchor failure even when the raw landmark CSV could support normalization.

This patch fixes the mismatch by:

- adding `133/362` to the default ALS oral-motor preset;
- keeping the old preset name as a backward-compatible alias;
- adding a clearer `ALS oral-motor core 17` preset name;
- adding `133/362` to lower-face presets where normalization is expected;
- adding `362` to the broad audit preset.

## Normalization fallback behavior

Default `intercanthal_distance` normalization now attempts:

1. inner canthus scale: `133/362`;
2. explicit outer-eye fallback: `33/263`;
3. failure with `normalization_anchor_missing` if neither pair is usable.

When fallback is used, the normalization manifest records:

- `scale_source = landmark_distance_33_263`
- `scale_status = fallback_outer_eye_anchors`
- `qc_flags` includes `normalization_anchor_fallback_used`

This makes normalization practical without hiding scientific uncertainty.

## Tests

Validated in the sandbox with:

```bash
python -m pytest tests/unit -k kinematics -q
PYTHONPATH=src python -m pytest tests/integration/test_acoustic_preprocess_stage.py -q
python -m compileall -q src/vslp/gui/kinematics/app.py src/vslp/analysis/kinematics/normalization.py src/vslp/analysis/kinematics/schemas.py src/vslp/analysis/kinematics/selection.py
```

Results:

- 46 kinematics unit tests passed.
- 1 integration test passed.
- Compile checks passed.
