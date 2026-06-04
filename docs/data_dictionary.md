# VSLP Data Dictionary

Minimum metadata columns for v1:

| Column | Meaning |
|---|---|
| `subject_id` | Stable patient/participant identifier. Required for leakage control. |
| `session_id` | Recording session identifier. Usually date/visit-level. |
| `iteration` | Longitudinal visit/session number. Not a repeated trial within a task. |
| `task` | Speech or kinematic task name. One file equals one task. |
| `recording_date` | Acquisition date. |
| `diagnosis` | Diagnostic group/class. |
| `severity_score` | Continuous severity score where available. |
| `severity_bin` | Ordinal/categorical severity grouping. |
| `file_name` | Original source file name. |

Recommended future column:

| Column | Meaning |
|---|---|
| `trial_id` | Repeated attempt of the same task within one session, if present. |
