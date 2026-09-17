# Acoustic manual review editor

The Segmentation stage keeps its automatic CSVs and static diagnostic PNGs under `acoustic/002_segmentation/`. The separate Segmentation manual review tab uses a live pyqtgraph waveform and a linked RMS panel. Mouse wheel and right drag zoom; middle drag pans. Click to seek, drag to select, and use **Play selection** to hear only the selection. The playback cursor and elapsed time follow Qt Multimedia playback. Display audio is min/max decimated for long recordings; all saved interval times remain full-precision seconds.

**Edit boundaries** reveals draggable speech regions and controls to add or delete speech, set the analysis start/end from the cursor, restore the full recording, mark a selected interval as contamination, undo, or reset to automatic. The optional text field accepts one `start_sec,end_sec` interval per line. Review edits stay in memory until **Keep automatic**, **Save manual**, or **Exclude recording** is selected. Reviewer identity and notes are required for corrections. The canonical WAV and automatic segmentation files are never rewritten.

Review state under `acoustic/003_segmentation_review/tables/`:

| File | Content |
|---|---|
| `segmentation_review_queue.csv` | Immutable automatic status, flags, paths, and summary SHA-256 |
| `segmentation_review_decisions.csv` | Decision, boundary source, analysis start/end, reviewer/date/notes, automatic paths |
| `manual_segmentation_overrides.csv` | Edited speech interval index/start/end with reviewer/date/notes |
| `reviewed_exclusion_intervals.csv` | Excluded start/end, reason, reviewer/date/notes |
| `review_entries/*.json` | Atomic per-record saved decision, overrides, and exclusions; replays after an interrupted CSV update |
| `final_segmentation_decisions.csv` | Frozen recording decisions and reviewed artifact paths |
| `final_segmentation_intervals.csv` | Frozen primary/strict interval views and complete timeline roles |

At freeze, the resolver intersects AUTO or MANUAL speech with `[analysis_start_sec, analysis_end_sec]`, subtracts reviewed contamination intervals, then reconstructs speech, leading nonspeech, internal nonspeech, trailing nonspeech, manual exclusion, and outside-analysis roles. Outside-analysis and manual-exclusion intervals are never physiological pauses. The final tables retain boundary source (`AUTO`, `MANUAL`, or `AUTO_MODIFIED`), reviewer/date, analysis window, exclusion reason, and traceable automatic interval index. Automatic `EXCLUDED` recordings can be rescued through review; computational `FAILED` recordings remain failures. QC and Features read the frozen intervals and final segment files while measuring the unchanged native-rate audio.

Qt Multimedia playback seeks to milliseconds; saved boundaries use independent full-precision seconds. Selection playback stops on media-position updates with a 20 ms timer, so audible stop timing depends on the system audio backend. The editor does not classify contamination automatically; the reviewer marks those regions.
