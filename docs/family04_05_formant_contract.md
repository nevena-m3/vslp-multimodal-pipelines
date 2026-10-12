# Families 04 and 05: reviewed formant contract

Family 04 reads native-rate canonical audio and frozen Alignment vowel tokens. It does not invoke MFA or alter the waveform. A recording whose native sample rate cannot support the selected Burg ceiling fails with `incompatible_native_sample_rate_formant_ceiling`; the ceiling is never lowered automatically.

## Frozen profiles

| Profile | Burg window | Step | Max formants | Pre-emphasis from | Ceiling |
|---|---:|---:|---:|---:|---:|
| `family04_burg_native_5500_v1` | 25 ms | 5 ms | 5 | 50 Hz | 5500 Hz |
| `family04_burg_native_5000_v1` | 25 ms | 5 ms | 5 | 50 Hz | 5000 Hz |

Vowel tokens shorter than 50 ms fail. Token values are medians of at least three finite, ordered F1/F2/F3 frames in the token's middle 50%. The 150–1200, 500–3500, and 1200–5000 Hz plausibility ranges flag values; they never clip them. Named-vowel values are medians of valid token medians. The optional sensitivity profile is an explicit user choice, never selected using demographic or inferred voice attributes.

## Mapping and artifacts

`docs/examples/vowel_category_manifest.example.json` is a schema example. Registered VSLP F04 tasks use the frozen versioned AA→/a/, IY→/i/, UW→/u/ mapping without project-by-project re-entry. An explicit mapping manifest remains a compatibility option. An invalid or unavailable mapping produces `missing_vowel_category_mapping` for named-vowel and derived outputs; token F1/F2/F3 remain measurable.

Family 04 writes `formant_token_measurements.csv`, `formant_vowel_measurements.csv`, and compact per-recording formant-frame NPZ files under `acoustic/005_features/tables/native_measurements/`. Token and named-vowel IDs are dimensions of those long-form tables; they are not collapsed into a recording-level scalar. Scalar /i,a,u/ outputs use named centroids and the exact formulas recorded in the registry. Output provenance includes the frozen Alignment run, segmentation hashes, mapping hash, native sample rate, Praat/Parselmouth versions, and profile ID.

Family 05 now has eight selectable outputs for the exact S2 buy/AY, S5 away/EY, and S5 ago/OW targets. It reuses the same native-rate F04 Burg tracker and frame QC, but samples the central 20–80% phone interval and uses a versioned robust bisquare line fit. S2 buy/AY alone also provides fitted-line F1/F2 range. The older NPZ reader and `docs/examples/formant_target_manifest.example.json` remain compatibility/research helpers; neither defines the production F05 target registry. See [Family 05 scientific review](family05_scientific_review.md) for exact outputs and source-parity limits.

The family documents refer to an older 48-kHz master and earlier segmentation padding. The current native-rate audio and frozen reviewed segmentation/Alignment contracts take precedence. Synthetic and imported alignment fixtures test the calculations; a live local MFA provider/model/dictionary run has not yet been validated.
