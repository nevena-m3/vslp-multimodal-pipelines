# Kinematics GUI v0.79 - Spacious Selection and Normalization Refinement

This patch is intentionally based on the restored v0.76/v0.77 state, not on the rejected compact v0.78 layout.

## Landmark Selection

- Keeps the spacious workstation model with a large real-video MediaPipe overlay.
- Uses a horizontal splitter between the control panel and the video canvas so the analyst can resize the workspace instead of accepting a cramped fixed layout.
- Keeps full-page scrolling available; the page is not compressed to avoid scrolling.
- Moves selected-landmark output, requirement checks, preset reference, and scientific notes into bottom tabs so each table can use the full width.
- Clips landmark drawing to the video frame region so zoomed points do not spill onto the dark canvas background.
- Adds a small zoom minimap when zoomed in. The inset shows the full frame and the current crop window, helping the user maintain spatial context.

## Normalization

- Removes the large schematic from the primary workflow.
- Keeps normalization explanations, but moves them into a secondary tab called "Why normalize".
- Uses a vertical, spacious workflow: readiness cards, policy controls, method audit/scientific notes, and output diagnostics.
- Keeps the method audit and method comparison available without crowding the main policy controls.

## Scientific behavior

No normalization mathematics, landmark IDs, MediaPipe extraction behavior, QC thresholds, or output schemas are changed by this patch. This is a layout and auditability refinement only.
