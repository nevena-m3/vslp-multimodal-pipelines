# VSLP Feature Analysis GUI v0.47.1 — Visual Selection Hotfix

This hotfix resolves a Windows-specific visual issue where combo boxes, selection controls, and plot buttons could appear dark/black before interaction.

Changes:
- Forced non-sidebar push buttons to use light backgrounds with navy text.
- Added explicit hover, focus, pressed, checked, and disabled states.
- Forced combo boxes and popup item views to use white/light backgrounds and readable dark text.
- Added selected and hover states for dropdown items.
- Preserved the left navigation sidebar styling while keeping all working controls visible.

This patch does not alter analysis logic or output files. It is a GUI readability hotfix only.
