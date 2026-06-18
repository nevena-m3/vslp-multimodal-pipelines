# Feature GUI v0.57 Reliability Empty Subject Fix

This patch hardens the Feature Analysis reliability screen for feature tables that contain a subject-like column but no non-missing subject identifiers.

Previously, the reliability subject-count table could raise `KeyError: 'n_records'` when no grouped subject rows were available. The GUI now returns an empty table with the expected schema and continues the analysis. This allows acoustic feature tables without linked subject metadata to proceed through column mapping and feature audit.

No ML training behavior is added.
