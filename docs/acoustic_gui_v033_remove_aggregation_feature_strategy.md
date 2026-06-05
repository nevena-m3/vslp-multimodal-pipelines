# VSLP v0.33 — Remove Aggregation and Move Reduction Logic into Feature Extraction

This update removes the Aggregation tab from the Acoustic GUI.

Rationale: the meaningful statistical decision is not a post-hoc group aggregation after a scalar feature table exists. The important decision is how each feature is reduced from its native measurement scale to one file-level value.

Feature Extraction now displays the computation strategy:

- timing/respiratory: segment-event summaries
- rhythm/EMS: effective-task envelope modulation preserving internal pauses
- phonatory: voiced-frame/period-track summaries
- articulatory/formant: valid speech-frame LPC trajectory summaries
- resonatory/nasality: valid spectral-frame median contrasts with warning flags
- coordination: aligned trajectory / lagged eigenspectrum summaries

The feature table remains one row per file. Future native-track persistence can be added for audit/research analysis, but the user-facing reduction decision belongs in Feature Extraction.
