# Acoustic Feature Formula Audit

**Audit date:** 2026-06-19  
**Authoritative inputs:** `Features Formulas (2).docx` and `Features Research (2).xlsx`

## Purpose

This document records how the acoustic extraction pipeline implements the supplied feature specification. It distinguishes mathematical definitions, signal-estimation procedures, reference ranges, and limitations that cannot be resolved from uncalibrated remote recordings alone.

## Implemented Corrections

| Area | Pipeline definition |
|---|---|
| Internal pause | Nonspeech interval inside the first-to-last speech interval with duration >= 300 ms. |
| Phrase | Continuous speech interval longer than 300 ms. |
| Pause and phrase CV | Sample standard deviation divided by arithmetic mean (`ddof=1`). |
| Percent pause | `100 * internal_pause_duration / effective_task_duration`; reported as percent, not proportion. |
| Intensity CV | Sample CV of 25-ms frame intensity in dB over speech-only audio, 10-ms hop. A nominal 20-uPa reference is used and the result is flagged as uncalibrated. |
| EMS peaks | Two strongest local spectral peaks in 0.5-10 Hz. |
| EMS band energy | Relative non-DC modulation power in 0-4, 4-10, and 3-6 Hz; denominator is total non-DC 0-10 Hz power. |
| EMS ratio | Energy 0-4 Hz divided by energy 4-10 Hz. |
| F1-F5 | Arithmetic mean over valid LPC frames. |
| F1-F5 bandwidth | Median root-derived bandwidth in Hz; `-fs/pi * log(root_radius)`. |
| F1-F3 range | Maximum minus minimum over valid frames, as specified. |
| F1-F3 slopes | Distribution of protocol-sign slopes `(F_start-F_end)/duration`, summarized by median, P5, P95, and P95-P5. |
| Jitter/shimmer | Praat-Parselmouth periodic PointProcess measures with period floor `0.8/Fmax`, ceiling `1.25/Fmin`, maximum period factor 1.3, and maximum amplitude factor 1.6. |
| Voice breaks | Number of adjacent PointProcess pulses separated by more than `1.25/Fmin`. |
| H1/H2 | Frequencies are exactly F0 and 2*F0; amplitudes are interpolated at those exact spectral locations. |
| DDK rate | Detected syllable nuclei divided by effective DDK task duration. |
| DDK regularity | Sample SD of inter-syllable intervals divided by their mean. |

## Source Corrections

The workbook contains several transcription defects. The implementation follows the DOCX formula where the two sources conflict:

- The workbook's `Ratio 0-4/4-10 Hz` row contains the CPP formula. The implemented ratio is EMS power from 0-4 Hz divided by power from 4-10 Hz.
- The bandwidth list labels F3 twice. The fourth item is interpreted as F4 bandwidth.
- The displayed percent-pause equation has an extraction-order artifact. The implemented numerator is pause duration and the denominator is effective task duration.
- Workbook ranges mix physical bounds, population references, pathology thresholds, and qualitative ALS directions. They are registry context only; values are not clipped or rejected solely because they fall outside those references.

## Measurement Limitations

- Consumer laptop WAV amplitude is not microphone-pressure calibrated. `RMSamp` is therefore reported in full-scale normalized amplitude, not Pascal. Intensity CV is appropriate for within-protocol variability work but must not be interpreted as calibrated SPL.
- Chen compensation functions for `A1P0comp` and `A1P1comp` are not fully specified in the supplied documents. These outputs remain explicitly marked `computed_proxy`; raw A1-P0 and A1-P1 are the primary auditable outputs.
- DDK nucleus detection is automatic and parameterized. Its event timestamps must be validated against manually annotated representative AMR and SMR recordings before clinical use.
- LPC formants remain sensitive to recording bandwidth, vocal-tract characteristics, vowel content, noise, and tracking failures. The pipeline retains validity provenance and does not treat the workbook's population ranges as universal rejection limits.
- Sex-specific F0 ranges and published jitter/shimmer thresholds are reference information, not universal diagnostic cutoffs.

## Verification

Formula-level tests use synthetic signals with known pauses, modulation frequency, formants, pitch cycles, voice interruption, and DDK event timing. The focused audit suite is maintained in:

- `tests/unit/test_timing_features_validated.py`
- `tests/unit/test_rhythm_features_validated.py`
- `tests/unit/test_articulatory_formant_features_validated.py`
- `tests/unit/test_phonatory_features_validated.py`
- `tests/unit/test_resonatory_features_validated.py`
- `tests/unit/test_ddk_features_spec.py`

These features are research outputs. Clinical interpretation requires task-specific validation, representative recording-condition validation, and locked preprocessing/version provenance.
