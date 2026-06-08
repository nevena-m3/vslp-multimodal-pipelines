# VSLP Feature Analysis GUI v0.55.1 — Export Completion Dialog Style Hotfix

This hotfix improves Windows readability for export-completion and other message dialogs.

## Change

Qt message boxes are explicitly styled with a white background, dark readable text, and visible light buttons across normal, hover, pressed, and focused states.

## Rationale

On some Windows Qt themes, modal notifications may inherit a dark native palette even when the main GUI is using the VSLP light theme. This can make export-complete text hard to read. The hotfix keeps notification dialogs visually consistent with the rest of the Feature Analysis GUI.

## Scope

No analysis logic, export files, plots, tables, or recommendations are changed.
