# Feature GUI v0.61 metadata-aware overview hardening

This patch hardens the Feature Analysis GUI after the v0.60 Overview redesign.

Changes:

- Visible version is bumped to v0.61.0.
- Source files are ASCII-safe so Windows default test reads do not fail on cp1252 decoding.
- Overview design plot is compact and status-aware rather than large tile cards.
- Detected-but-empty design columns are labelled as empty columns, not usable metadata.
- Metadata aliases are normalized for common exports, including `Raw Media File name`, `SubjectID`, `Task Name`, `Diagnosis`, `Sex`, `Recording date`, and `ALSFRS total score`.
- Metadata can join by exact filename/basename/stem, then record/session/task style keys.
- Empty canonical feature-table columns such as `subject_id`, `task`, or `diagnosis` can be filled from matched metadata while preserving explicit `metadata__...` columns for transparency.

Boundary: this does not implement full filename parsing for unmatched kinematic/video IDs. That remains the next planned patch.
