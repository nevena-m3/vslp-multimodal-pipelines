# VSLP Acoustic GUI v0.14 Product Design Fix

This release corrects the v0.13 visual/layout regression.

## What changed

- Removed fragile QLabel background styling that created black text artifacts on macOS.
- Replaced the over-styled dark theme with a restrained clinical/research product theme.
- Reduced sidebar width and stage-card height.
- Reduced banner/log vertical footprint.
- Made major tabs scrollable so action buttons remain reachable on laptop screens.
- Reduced feature-detail and plot-preview minimum heights.
- Improved plot preview scaling for smaller windows.
- Preserved scientific rationale and tooltips without overloading the UI.

## Product design principle

The GUI should feel like a professional research instrument: calm, readable, technically clear, and robust on real laptop displays. Visual treatment should support workflow execution, not compete with it.
