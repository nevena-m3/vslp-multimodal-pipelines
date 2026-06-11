# Feature GUI v0.77 - Project filename template parser

Built from v0.76 QC integration repair.

## Adds to Project
- Shows one real example filename/stem after a feature table is loaded.
- Splits the example into tokens.
- Lets user assign token positions for:
  - subject_id
  - protocol_id
  - iteration
  - recording_date
  - task_code
  - task
- If all fields remain Auto, the existing safe VSLP-style filename parser is used.
- If any field is assigned, the user template is applied consistently across rows.

## Safety
- Metadata still has priority where successfully matched.
- Diagnosis/severity are not inferred from filename.
- If no metadata exists, filename context can still provide subject/iteration/date/task/task_code.
- No global filter and no analysis mutation beyond context columns already used by the Feature GUI.
