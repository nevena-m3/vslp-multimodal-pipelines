# GUI Design Direction

The VSLP GUI suite should feel like professional research instrumentation software, not a notebook wrapper.

Design direction:

- Navy/dark visual system.
- Project-first workflow.
- Clear left navigation by stage.
- Central parameter panel with clinical/simple mode and expert mode.
- Right-side provenance/status panel.
- Bottom log/error console.
- Per-stage plots, tables, and reports.
- Batch progress and file-level failure handling.
- No silent overwrites.

Qt/PySide6 is selected for v1 because it supports native local file/folder selection, long-running worker threads, professional desktop layout, and packaging for macOS/Windows.
