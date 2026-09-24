# Family 06 segmental contrast contract

Family 06 consumes frozen reviewed segmentation and frozen Alignment. Phone identity
and timing come from the Alignment API. Burst and noise landmarks require approved
external annotations on the original recording clock. A phone boundary is never
silently substituted for an acoustic sub-event. The Family 06 stage does not run
MFA, ASR, segmentation, or a burst detector.

The version 1 target manifest identifies the exact task, word, phone, token indices,
aggregation, and any matched pair. The validated sub-event CSV supplies target-ID
matched burst/noise times, annotation source, reviewer, approval, and original-time
axis. Empty or mismatched resources produce per-feature NaN and a reason. They do
not prevent unrelated selected families from completing.

The public executable leaves are `m1_t_minus_k_hz`,
`wideband_noise_energy_0_10khz`, and `stop_burst_spectral_tilt_db_khz`.
Their formulas and fixed source windows follow the Family 06 document. The named
`family06_native_subevents_v1` profile records these engineering choices where
the source gives no exact FFT normalization: at least 2048 FFT points, Hann
window, FFT power from digital amplitude, one-sided Parseval energy divided by
the window mean-square for the 0–10 kHz sum, and 20 log10 magnitude with a
relative 1e-12 numerical floor for burst tilt. These choices are reproducible;
source or clinical parity has not been established. The 0–10 kHz measure is
channel sensitive digital energy, never calibrated physical energy. Native
Nyquist at or below 10 kHz fails with `insufficient_bandwidth`; no upsampling
can restore source bandwidth.

`blocked_aural_analytics_ap` is proprietary. The two band-noise contrast variants
lack a complete source contrast/normalization and target-pairing contract. The
`noise_duration_<target>_s`, `noise_rise_time_<definition>_s`, and
`normalized_duration_contrast_<A>_<B>` IDs are unresolved templates. Generic
duration and normalized-contrast calculators are private utilities only.
The missing noise rise endpoints are deliberately not guessed.

Each Family 06 run writes `family06_segmental_targets.csv` with alignment and
target IDs, word/phone indices, phone and measurement bounds, annotation source,
reviewer, token value, validity, failure reason, parameter set, and algorithm
version. The stage manifest stores hashes of the target and annotation inputs.
The unified Acoustic Features result stores one scalar/status row per selected
recording and feature.

The Family 06 document's older 48-kHz master and segmentation guidance is
superseded by the current native-rate canonical audio, frozen reviewed
segmentation, and frozen Alignment contracts.
