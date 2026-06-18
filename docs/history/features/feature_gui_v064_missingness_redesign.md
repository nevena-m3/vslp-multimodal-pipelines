# Feature GUI v0.64 - Missingness redesign

This patch redesigns the Missingness menu to match the Overview page layout pattern: a large plot-first review area, a compact right-side snapshot panel, and detailed tables below.

The change is intentionally UI-only. It does not change feature calculations, missingness statistics, metadata joining, QC, or ML export logic.

Missingness now keeps only availability-specific plots:

- Feature missingness burden
- Recording missingness burden
- Missingness by metadata group
- Missingness by feature family
- Feature availability heatmap
- Co-missingness clusters

Distribution shape, QC sensitivity, outcome screening, feature relationships, and ML-export readiness remain in their own menus.
