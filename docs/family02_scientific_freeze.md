# Family 02 scientific and software freeze

Group 2 contains nine constructs: jitter, shimmer, HNR, GNE, PVI, CPP, DFP, H1–H8 harmonic structure, and H1–H2 harmonic difference. The calculations are engineering tested. Independent source-package parity, real VSLP recording validation, and clinical validity remain pending. No result is a validated clinical threshold.

## Task and input contract

| Construct | Sustained /a/ | S5 | S6 | Bamboo |
|---|---|---|---|---|
| Jitter, shimmer, HNR, DFP, H1–H8 | recommended | recommended | not recommended | not recommended |
| GNE, PVI | recommended | not recommended | not recommended | not recommended |
| Unsmoothed CPP | recommended | recommended | recommended | recommended |
| H1*–H2* corrected | recommended | not recommended | not recommended | not recommended |
| H1–H2 raw comparison | secondary, not recommended | not recommended | not recommended | not recommended |

Recommendation is separate from executability. S5 is “We were away a year ago”; S6 is “My mama makes lemon jam”. F02 uses upstream task identity and final reviewed segmentation, without requiring MFA. Sustained /a/ uses the reviewed stable region's centered interval of at most 2 seconds. Other task regions retain original timestamps and hard manual exclusions. The canonical acoustic waveform retains its native sample rate; at least 16 kHz is required. The 16-kHz VAD sidecar is never the source of these measurements. GNE alone resamples its selected input internally to 8 kHz with polyphase anti-aliasing.

Each feature is calculated from trial-level primitive observations. Distinct voiced islands do not create cross-gap jitter, shimmer, DFP, or harmonic windows. Trial-level HNR, CPP, and GNE pool valid frames; harmonic statistics normalize against a single maximum across all valid trial windows. The preferred recording value is the median of valid trial values. Individual trial measurements and pulse times remain in native audit tables.

## Source-derived definitions and settings

- Jitter: Praat periodic cross-correlation pulse track, 60–500 Hz, valid periods 0.0016–0.02083 s, maximum period factor 1.3. Local, absolute, RAP, PPQ5, and DDP remain separate.
- Shimmer: the same pulse settings and Praat period AmplitudeTier; maximum amplitude factor 1.6. Local %, local dB, APQ3, APQ5, APQ11, and DDA remain separate.
- HNR: Praat Harmonicity cross-correlation, minimum pitch 60 Hz, 10-ms step, silence threshold 0.1, 4.5 periods/window; mean valid frame dB values. It is harmonic-to-noise, not NHR.
- GNE lineage: 8-kHz signal; 30-ms frames every 10 ms; 10th-order LPC autocorrelation with Hamming window and inverse-filter residual; 500/1500/2500-Hz band centers with 1000-Hz bandwidth; Hilbert envelopes and cross-band correlation maxima. `gne_mean` and `gne_sd` summarize frame values. `gne_source_replication_only` is a deprecated alias of `gne_mean`.
- PVI lineage: 5-ms F0 contour; mean-normalized F0; third-order Butterworth 9–14-Hz bandpass; Welch 1-second window with 95% overlap; sum 9–14-Hz spectral amplitudes.
- CPP: unsmoothed Praat PowerCepstrogram → PowerCepstrum slice → Get peak prominence; 60–500-Hz peak search, parabolic interpolation, 0.001–0.05-s straight baseline, robust-slow fit. Parselmouth 0.4.7 embeds Praat 6.1.38. This is CPP; CPPS stays unavailable.
- DFP: count nonzero-sign reversals in adjacent fundamental-period differences within valid runs. The original [Askenfelt and Hammarberg (1981)](https://www.speech.kth.se/qpsr/1981/1981_22_4_049-068.pdf) describes the denominator as the total number of adjacent-period differences, **N−1** per run. The earlier local F02 DOCX states N−2; that conflicts with this original source. Zero differences do not reverse direction. S5 use is a VSLP task extension.
- H1–H8 lineage: eight-period frames with one-period overlap (seven-period advance), interpolate to 4096 samples, Hamming window, first eight harmonic bins, global trial amplitude maximum, `20 log10(h/global_max)`, sample SD, and `RelH=1/(abs(mean)+SD)`. The 24 scalar leaves remain distinct.

## Explicit VSLP replication choices

- GNE filter family/order: fourth-order Butterworth SOS; 0–1000-Hz lowpass and 1000–2000/2000–3000-Hz bandpasses. Cross-envelope lag search ±1 ms. These details were not recovered from the source material. The LPC regularization and frame-edge convention are also VSLP choices.
- PVI internal gaps: interpolate only up to 20 ms, reject over 10% missing F0 and edge gaps. Welch uses SciPy `scaling="spectrum"`, a Hann window, and square root of one-sided spectral power per bin before summing bins. This amplitude normalization is a versioned VSLP convention pending exact source parity.
- H1–H8 period landmarks come from the VSLP Praat periodic-cc pulse estimator. Source estimator parity remains pending.
- Native sample-rate minimum, centered /a/ interval, S5/S6 recommendation, and median of valid trials are VSLP lab/software decisions.

All custom IIR designs use checked second-order sections. [Filter-response QA](qa/family02/family02_filter_responses.png) and [coefficients/metadata](qa/family02/family02_filter_metadata.json) document the frozen design and pole-radius checks. These artifacts are engineering QA, not feature or clinical results.

## Ninth construct: H1*–H2* formant-corrected harmonic difference

The preferred `h1star_h2star_db` corrects the first and second harmonic for the boosts caused by F1 and F2; the secondary `h1_h2_db_raw` is the uncorrected H1−H2 comparison. Both are project-defined leaves of C080, an approved addition after the original 79-construct matrix. They are distinct and selectable. Only sustained /a/ is recommended for the corrected measure; S5, S6, Bamboo, and DDK remain technically executable but are not recommended. Raw H1−H2 is secondary for every task.

For `ω=2πf/Fs`, `rᵢ=exp(−πBᵢ/Fs)`, and `θᵢ=2πFᵢ/Fs`, the [Iseli, Shue, and Alwan (2007) equation 1/A5](https://www.seas.ucla.edu/spapl/paper/Iseli_JASA_07.pdf) specifies the formant boost:

`boost(f) = Σᵢ₌₁² 10 log₁₀[(1 − 2rᵢcos θᵢ + rᵢ²)² / {(1 − 2rᵢcos(ω+θᵢ)+rᵢ²)(1 − 2rᵢcos(ω−θᵢ)+rᵢ²)}]`.

The reported corrected value is `(H1_dB − boost(F0)) − (H2_dB − boost(2F0))`. The estimator uses F1/F2 and B1/B2 at the matched frame time. It prefers measured Praat Burg bandwidths when they meet the versioned physical checks (`20–1000 Hz`, below the formant frequency); otherwise it records `formula_fallback` and uses the Iseli source-discussed `Bᵢ=80+120Fᵢ/5000` Hz formula. Zero bandwidth is never a default. The existing F04 native-rate Burg configuration supplies formants and bandwidths directly from reviewed audio; F02 does not require F04 execution or Alignment.

Harmonics use 25-ms Hann frames at shared 5-ms F0 timestamps, an 8192-point FFT, and linearly interpolated magnitude at F0 and 2F0. The reviewed stable /a/ region remains centered and at most 2 seconds. The frame median is the trial value after at least 20 valid frames; the median of valid trial values is the recording result. The frame audit retains original time, F0, H1/H2, F1/F2, B1/B2, bandwidth source, both outputs, and QC flags. Invalid formants make the corrected value unavailable without silently substituting the raw result. Extreme corrections are flagged without clipping. The 20-frame and bandwidth credibility rules are technical VSLP settings, not clinical cutoffs.

H1*–H2* is an acoustic correlate of glottal-source spectral shape, not a direct physiological measurement. F0, vowel/formant configuration, age, sex-related physiology, and channel frequency response affect interpretation. Extraction uses no sex normalization, universal clinical threshold, or uncalibrated microphone correction. Independent VoiceSauce/Iseli numerical parity, real VSLP data checks, and clinical validation remain pending.

## Remaining parity work

Compare jitter/shimmer and CPP against independent Praat reference exports; source GNE against an original implementation including band-filter and lag conventions; PVI spectral-amplitude scaling against a source reference; DFP on a source reference sequence; and H1–H8 against a reference pulse estimator. Test on consented real VSLP recordings separately. Do not interpret passing synthetic tests as clinical validation.
