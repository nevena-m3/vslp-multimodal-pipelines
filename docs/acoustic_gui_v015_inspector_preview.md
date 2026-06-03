# VSLP Acoustic GUI v0.15 - Inspector Preview Layout

This release focuses on the Inspector tab only.

## Product design change

The Inspector now uses a two-column layout:

- left rail: table and plot selection buttons
- right canvas: large table or near-square plot preview

This makes plots substantially easier to inspect on laptop screens and keeps the workflow consistent with a professional analysis product.

## Plot preview behavior

Generated PNG plots are not re-rendered inside the GUI. The GUI loads the already-generated audit artifact from disk and displays it in a near-square preview canvas while preserving aspect ratio.

This avoids mixing backend plotting with Qt rendering, which is safer on macOS and keeps the output artifact identical to the report file saved on disk.

## Current preview file

The Inspector now includes **Open current preview file**, which opens the currently selected table or plot in the native macOS application.
