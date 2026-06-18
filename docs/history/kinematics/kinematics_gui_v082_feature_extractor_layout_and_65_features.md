# Kinematics GUI v0.82 — Feature extractor layout and 65-feature scalar layer

## Purpose

This patch refines the Features menu after v0.81. It has two goals:

1. Make the feature selector/extractor the dominant working area of the tab.
2. Implement the uploaded 65-feature orofacial kinematic feature-map scalar layer explicitly and audibly.

The implementation remains research-grade. It does not add ALS, Parkinson's disease, or other diagnostic cutoffs.

## GUI changes

The Features tab is reorganized as follows:

- top: compact feature status cards;
- main working area: a large feature selector/extractor tree on the left, with run settings and interpretation on the right;
- bottom: tabbed tables for computed features, feature framework, implementation audit, and QC requirements.

The selector is intentionally wide and visually central. Framework and audit tables no longer compete with the selector at the top of the page.

## Computation changes

The feature backend now emits the canonical 65 scalar feature names represented in the uploaded feature map:

- vertical opening path and ROM: `path_vert_*`, `rom_vert_*`;
- vertical speed and acceleration: `sLL_vert_*`, `aLL_vert_*`;
- horizontal spreading path and ROM: `path_horz_*`, `rom_horz_*`;
- horizontal speed and acceleration: `sLL_horz_*`, `aLL_horz_*`;
- mouth aspect ratio: `aspect_*`;
- jaw lateralization ratio: `jaw_lat_*`;
- lip symmetry ratio: `lip_symm_ratio_*`;
- bilateral commissure coordination: `lat_xcorr`.

The feature families are computed from normalized landmark trajectories. The current GUI normalization convention uses inner canthus landmarks `133/362` with fallback `33/263`. The uploaded feature map also uses `243/463` as canthus references for some legacy lateralization/symmetry ratios when those landmarks are present.

## Accuracy decisions

The known vertical/horizontal naming collision described in the uploaded map is fixed by using disambiguated names such as `sLL_vert_med` and `sLL_horz_med`.

The old inconsistent `*_prc_5_95_iqr` definition is not reproduced. v0.82 uses a consistent, non-negative IQR of movement-window ranges for the `*_prc_5_95_iqr` family.

Cumulative path is computed as the sum of absolute frame-to-frame change in normalized position within each movement window. This gives normalized-distance units and avoids frame-rate-dependent index integration.

## Backward compatibility

The time-series layer still writes support signals such as `mouth_aperture`, `mouth_aperture_velocity`, `outer_lip_spread`, and `lip_aspect_ratio` so existing aggregation and tests continue to work.

## Validation

Validated in the development sandbox with:

```text
54 kinematics unit tests passed
1 acoustic integration test passed
compile checks passed
```
