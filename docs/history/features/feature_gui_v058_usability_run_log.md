# Feature GUI v0.58 usability and run log

This patch hardens the Feature Analysis GUI user interface after v0.56/v0.57.

## Changes

- Bumps the visible Feature Analysis GUI version to `v0.58.0`.
- Adds a persistent bottom run log to the main Feature GUI window.
- Adds a progress bar that updates during load, mapping, analysis, and export milestones.
- Improves scroll-area behavior for wide feature-analysis pages.
- Improves table readability with pixel scrolling, interactive column resizing, alternating rows, and resize-to-contents behavior.
- Adds error logging to the persistent run log for load and analysis failures.

## Boundary

This is a usability patch only. It does not alter feature calculations, feature recommendations, ML exports, or model training. Subject/session/task inference without metadata is intentionally deferred to a separate patch.
