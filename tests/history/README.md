# Historical Tests

This directory preserves implementation-era source snapshots that describe earlier GUI releases. They are retained as engineering provenance and are intentionally excluded from the default pytest collection.

Historical snapshot tests often assert an exact past application version, layout string, or transitional behavior. Running them against the current GUI is expected to fail and does not represent a current regression.

Current regression tests live under `tests/unit`.

When a historical behavior remains scientifically or operationally important, restate it as a version-independent current test under `tests/unit` rather than reactivating the old snapshot.
