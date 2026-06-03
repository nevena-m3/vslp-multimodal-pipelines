# Feature Region Policy

Many acoustic biomarkers are region-dependent. VSLP v0.12 therefore makes the analysis region explicit.

## Why full-file extraction is often wrong

Remote clinical recordings contain leading silence, task instructions, hesitations, internal pauses, and trailing silence. Computing signal features over the full file can contaminate phonatory, spectral, rhythm, and amplitude estimates.

## Current VSLP policy

Timing/pause features are computed from Silero `segments_df`.

Signal features default to `speech_only`, which concatenates Silero speech regions before computing phonatory, rhythm, and baseline resonatory/intensity features.

Expert users may choose `effective_task` or `full_file` in the GUI.

## Future work

Some features should eventually use task-specific regions:

- sustained vowels: stable voiced vowel core;
- DDK: syllable bursts and inter-burst intervals;
- connected speech: speech regions excluding long pauses;
- pause biomarkers: internal nonspeech only;
- resonance/formant features: voiced sonorant/high-energy speech frames.

These will be implemented as validated plugins.
