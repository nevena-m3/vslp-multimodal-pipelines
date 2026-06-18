# VSLP Acoustic GUI v0.20 - Data Segmentation refinement

This update turns the segmentation block into a method-selection screen rather than a Silero-only screen.

## Implemented now

- Silero VAD speech/pause segmentation.
- Method selector in the GUI.
- Reserved plugin slots for SPA, Energy/RMS, and Custom segmentation.
- Compact user-facing segmentation summary table.
- GUI segmentation summary after the stage runs.

## Method guidance

Silero is recommended for passages, sentences, and free speech where the main target is speech activity versus nonspeech/pause regions.

SPA and other task-specific methods are reserved for DDK or syllable/event segmentation workflows. These methods are visible in the GUI as intentional placeholders but are not implemented yet.

## Output tables

The segmentation stage now writes:

- `acoustic_segmentation_summary.csv`: detailed downstream table.
- `acoustic_segmentation_main_summary.csv`: compact user-facing summary.
- per-file frame tables.
- per-file segment tables.
- per-file boundary tables.
- error table.

The detailed table remains the downstream source of truth for feature extraction.
