# VSLP Data Contract and Dictionary

## Core Principle

Every row must have a documented unit and be traceable to source media. Column roles must be explicit before Feature Analysis or machine learning.

## Canonical Context Fields

| Column | Role | Definition |
|---|---|---|
| `record_key` | Identifier | Preferred unique key for one source recording/analysis row |
| `subject_id` | Identifier | Stable participant identifier; required for subject-grouped ML splitting |
| `session_id` | Identifier/context | Recording session identifier |
| `visit_id` | Identifier/context | Clinical/research visit identifier when distinct from session |
| `iteration` | Context | Longitudinal visit/session order; not a repeated trial unless defined that way |
| `trial_id` | Context | Repeated attempt of the same task within a session, when present |
| `task` | Context | Canonical task name represented by the recording |
| `task_code` | Context | Source or protocol task code |
| `recording_date` | Context | Acquisition date in an unambiguous date format |
| `file_name` | Provenance | Original source filename |
| `source_path` | Provenance | Local source lineage; do not expose outside approved environments |
| `modality` | Context | `acoustic` or `kinematic` |

## Analysis Roles

| Role | Meaning | Default ML treatment |
|---|---|---|
| Identifier | Row, recording, subject, visit, or file key | Never a predictor |
| Feature | Extracted measurement intended for scientific review | Candidate predictor after task-specific audit |
| Target | Outcome, diagnosis, label, or endpoint to predict | Response only; never a predictor |
| Covariate | Prespecified adjustment or stratification variable | Use according to protocol inside validation |
| QC | Recording/extraction quality evidence | Filtering/sensitivity/covariate role; not a disease predictor by default |
| Provenance | Path, software, method, status, or processing lineage | Never a predictor |
| Ignore | Administrative or unsupported field | Excluded |

## Common Clinical/Study Fields

| Column | Typical role | Notes |
|---|---|---|
| `diagnosis` | Target or grouping variable | Requires approved coding dictionary |
| `severity_score` | Target/outcome | Continuous clinical measure |
| `severity_bin` | Target/group | Derived ordinal/categorical severity grouping |
| `age` | Covariate | Define age reference date |
| `sex` / `gender` | Covariate/group | Preserve study terminology and coding provenance |
| `site` | Covariate/group | Important for multi-site/device effects |
| `device` / `microphone` / `camera` | QC/covariate | Potential technical confound |

## Missing Values

Missing values should remain missing, not be encoded as zero or a clinical category. Use a consistent representation in exported CSV files. Record why measurements are missing where status fields are available.

## Uniqueness and Alignment

Validate the intended key before merging. Preferred order for multimodal alignment:

1. `record_key`
2. `subject_id + session_id + task`
3. `subject_id + visit_id + task`
4. `subject_id + task` only when uniqueness is proven

Never force a many-to-many merge. Produce unmatched and excluded-row reports.

## Naming and Types

- Use stable `snake_case` canonical names in new machine-readable outputs.
- Preserve source labels in provenance/mapping tables.
- Store identifiers as strings, including identifiers that look numeric.
- Use ISO 8601 dates where possible.
- Record feature units, family, scale, and implementation status in registries/manifests.
- Do not embed target meaning in opaque numeric codes without a coding dictionary.
