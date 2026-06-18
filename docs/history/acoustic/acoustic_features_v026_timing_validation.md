# VSLP v0.26 — Respiratory/timing feature validation

This update validates the first acoustic feature subsystem: respiratory/timing.

## Why this group first

Respiratory/timing features are derived from the segmentation table rather than from fragile F0, LPC formant, CPP, or nasality estimates. They are therefore the safest first subsystem to validate.

## Effective task interval

The effective analyzed duration is defined as:

```text
first speech onset → last speech offset
```

This removes leading and trailing nonspeech while preserving internal pauses.

If legacy segment tables do not include timestamps, the fallback is:

```text
total duration - leading nonspeech - trailing nonspeech
```

## Validated formulas

```text
total_dur = last_speech_end - first_speech_start
speech_dur = sum(duration of speech segments)
total_pause_dur = sum(duration of internal nonspeech segments >= minimum_internal_pause_sec)
percent_pause = 100 * total_pause_dur / total_dur
num_pause = count(internal nonspeech segments >= minimum_internal_pause_sec)
mean_pause_dur = mean(internal pause durations)
mean_phrase_dur = mean(speech segment durations)
cv_pause_dur = std(internal pause durations) / mean(internal pause durations)
cv_phrase_dur = std(speech segment durations) / mean(speech segment durations)
speech_rate = 60 * known_task_word_count / total_dur
```

## Important policy decisions

`percent_pause` is now stored as percent units, not a 0–1 proportion.

`mean_pause_dur` is `NaN` when no valid internal pause exists. This is more honest than writing `0`, because the mean of an empty set is undefined.

`cv_pause_dur` and `cv_phrase_dur` are `NaN` when fewer than two observations are available.

`speech_rate` is only computed when a task word count is explicitly supplied. The pipeline does not infer word count from task name unless configured.

## Clinical interpretation

These features quantify connected-speech timing, breath-grouping, internal pausing, and speech/pause allocation. They should not be interpreted as direct respiratory physiology, but they are meaningful acoustic correlates of speech timing burden and bulbar progression monitoring.

## Limitations

Silero VAD provides speech/pause regions, not syllable nuclei. Therefore this implementation does not compute syllable rate or articulation rate unless a future SPA/DDK/task-specific segmentation module supplies the required counts.
