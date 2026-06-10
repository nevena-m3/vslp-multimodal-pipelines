# Feature GUI v0.62 menu plot standardization

This patch standardizes the Feature Analysis GUI review menus so that plot controls follow the same pattern introduced in the Overview redesign.

## Scope

- Bumps visible Feature GUI version to `v0.62.0`.
- Adds a shared plot-gallery builder for review pages.
- Uses a consistent toolbar pattern: plot dropdown, Show, Regenerate, and Open current plot on the far right.
- Keeps plots above detailed tables.
- Reduces visible plot redundancy by assigning each menu a focused plot list.
- Does not change feature computations, QC calculations, missingness calculations, model export logic, or ML training logic.

## Menus covered

- Missingness
- Distributions / Outliers
- QC Integration
- Feature Relationships
- Group / Outcome Screening
- Reliability / Repeatability
- Feature Recommendation

Overview remains the reference layout.
