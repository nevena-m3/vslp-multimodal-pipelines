# Family 12 — spectral and cepstral descriptors

## Status and task contract

Seven constructs (C065–C071) are defined. There are **130 selectable task-specific leaves**. Two C068 openSMILE leaves are documented as unavailable pending exact source configuration; no generic band-energy substitution is used. Scientific definitions are FROZEN except C068 source configuration. Implementation is FROZEN for active leaves; focused engineering has passed for numerically exercised paths. C067 numerical execution is PENDING in the current environment because librosa is unavailable. C068 is RESEARCH PENDING SOURCE CONFIG and nonselectable. These statuses do not establish reference, real-data, or clinical validation.

| Construct | Tasks | Public leaves |
| --- | --- | ---: |
| C065 MFCC | sustained /a/, S2, S6 | 78 |
| C066 bandwidth | Bamboo | 2 |
| C067 contrast | sustained /a/, S5 | 14 |
| C068 openSMILE 1–4 kHz percentile 1 | S1/WSTG, S7 | 0 (2 unavailable) |
| C069 ZCR | S1/WSTG, S7 | 4 |
| C070 spectral moments | S1/WSTG, S2, S7 | 12 |
| C071 spectral shape | S1/WSTG, S2, S7, Bamboo | 20 |

The registered WSTG task is the S1 task for this feature map. Generic historical IDs are nonselectable. S1/S2/S7 C070 owns public centroid, spread, skewness and kurtosis; C071 exposes only slope, flux, crest and flatness there. Bamboo C071 exposes all eight shape descriptors. C070 and C071 use the same low-level moment arrays.

## Authority, audio, and trials

The input is final reviewed segmentation and, for sentence/passage tasks, a human-confirmed current-hash Reviewed Trial Confirmation V1. The sustained vowel uses the neutral reviewed stable-phonation helper and its centered span of at most 2.0 s. Sentence and Bamboo measurements use eligible reviewed speech within each trial. Hard exclusions are removed before framing. Each continuous interval is resampled and framed independently; no STFT or flux transition crosses a gap, exclusion, or trial. Trial summaries are computed first; recording summaries are medians across valid trials. Final Alignment, MFA, and TextGrid are not prerequisites.

The source/native waveform remains untouched. The private analysis copy uses 48,000 Hz with scipy.signal.resample_poly; native rate, analysis rate, resampler and SciPy version are in trial audit. The common transform is a 25 ms symmetric NumPy Hann window (1200 samples), 10 ms hop (480 samples), FFT length 2048, center=False, no padding, denoising, or peak normalization. At least 10 valid frames are required.

## Definitions

C065 uses 40 Slaney-area-normalized mel bands from 50–8000 Hz, natural log, orthonormal type-II DCT, first 13 coefficients including C0, no pre-emphasis. Each coefficient yields a frame mean and sample SD (ddof=1) per trial. Coefficients are cepstral coordinates, not individual physiological measurements. The emitted mfcc01 through mfcc13 map to mathematical DCT C0 through C12, respectively. C0 is the orthonormal sum of log mel-band powers and carries substantial overall-level information, although it is not a separate total-energy coefficient. The S2-versus-S6 oral-nasal composite belongs to downstream Feature Analysis. Downstream validation must compare a full MFCC-vector difference including C0 with a spectral-shape difference excluding the energy-sensitive C0, and assess coefficient scaling or standardization before choosing a Euclidean distance. Coefficient-wise signed S6-minus-S2 deltas may remain support values; no definitive composite is frozen here.

C066 uses STFT **magnitude** weights: centroid = sum(Sf)/sum(S); BW2 = sqrt(sum(S(f-centroid)^2)/sum(S)). Public Bamboo outputs are frame mean and frame median in Hz. This is not occupied-power bandwidth.

C067 uses the declared librosa 0.11-style spectral_contrast configuration: fmin 200 Hz, six octave divisions yielding seven bands, quantile 0.02, dB peak-to-valley output. The highest band extends to the 24 kHz Nyquist limit. Production calls `librosa.feature.spectral_contrast` on the magnitude STFT (frequency-by-frame), with `fmin=200`, `n_bands=6`, `quantile=0.02`, and `linear=False`. The current local .venv-win lacks the declared librosa package; affected values are unavailable with `librosa_unavailable`. Its scientific definition is frozen and implementation is defined, but this local numerical path has not been independently exercised.

C068 historically names `pcm_fftMag_fband1000-4000_sma_percentile1.0`. The repository does not contain the exact openSMILE feature configuration, smoothing, scale and percentile pipeline. Its S1/S7 leaves remain nonselectable with `research_pending_source_config`.

C069 subtracts each frame mean, then divides adjacent sign-change count by the number of adjacent pairs. Public frame mean and median are fractions 0–1. It is highly noise-sensitive.

C070 uses normalized **power** weights p=P/sum(P). M1=sum(pf) Hz; M2=sqrt(sum(p(f-M1)^2)) Hz; M3=sum(p(f-M1)^3)/M2^3; M4=sum(p(f-M1)^4)/M2^4. M4 is raw Pearson kurtosis, with no subtraction of three. Public aggregation is frame median.

C071 slope uses the STFT magnitude spectrum at positive-frequency bins only, **>0 Hz through 24 kHz** (DC excluded, Nyquist retained). For each valid frame, let `A_ref` be the finite positive maximum magnitude over those bins and set `L_k = 20log10(max(A_k/A_ref, 1e-6))`. The frozen floor is -120 dB relative to that frame maximum; levels span -120 to 0 dB. Fit unweighted ordinary least squares `L_k = beta0 + beta1*(f_k/1000)`; equivalently `beta1 = sum((f_k_khz-mean(f_k_khz))*L_k)/sum((f_k_khz-mean(f_k_khz))^2)`. The frame output is signed `beta1` in dB/kHz; the trial value is the median valid frame slope and the recording value is the median valid trial values. Adding a constant dB level does not change the theoretical OLS slope, so frame-relative normalization removes absolute gain from the intercept while preserving the slope except where the deliberate -120 dB floor applies. This frequency band is a **VSLP operationalization**, not a documented source band; source-band parity remains pending. Flux is the L2 distance between consecutive unit-L2 magnitude spectra inside each continuous interval only. Crest=max(magnitude)/mean(magnitude). Flatness=geometric mean(power)/arithmetic mean(power), replacing zero power bins with `numpy.finfo(float).tiny` only inside the logarithm. Public values are medians across valid frames or valid within-interval flux pairs.

## Spectral representation audit

| Construct / output | Representation | Weighting or transform | Public aggregation |
| --- | --- | --- | --- |
| C066 bandwidth | Hann-windowed real-FFT magnitude | Magnitude-weighted centroid and second central moment (p=2) | Frame mean and median |
| C067 contrast | Magnitude STFT, passed as `S=good_mag.T` to librosa | Octave-band 2% peak/valley quantiles, log-ratio dB | Per-band frame mean |
| C068 openSMILE band percentile | Historical `pcm_fftMag` 1–4 kHz lineage | Exact smoothing, scale, and percentile pipeline unresolved | Unavailable |
| C069 zero-crossing rate | Time-domain, frame-mean-subtracted signal | Adjacent sign changes divided by adjacent-pair count | Frame mean and median |
| C070 M1–M4 | Power spectrum | Normalized power weights `P/sum(P)`; M4 is raw kurtosis | Frame median |
| C071 centroid/spread/skew/kurtosis (Bamboo) | Power spectrum | Same normalized power moments as C070; raw kurtosis | Frame median |
| C071 slope | Frame-relative FFT magnitude, `20log10(max(A/A_ref, 1e-6))` | Unweighted OLS against kHz over >0 Hz through Nyquist; -120 dB relative floor | Frame median |
| C071 flux | Unit-L2-normalized magnitude spectra | L2 distance of consecutive valid frames within one continuous interval | Valid-pair median |
| C071 crest | FFT magnitude | Maximum magnitude / mean magnitude | Frame median |
| C071 flatness | Power spectrum | Geometric mean / arithmetic mean; tiny floor inside logarithm | Frame median |

C070 publicly owns M1–M4 for S1/S2/S7; C071 exposes only slope, flux, crest, and flatness for those tasks. Bamboo C071 exposes all eight descriptors. Flux never forms a pair across reviewed interval, pause, hard-exclusion, trial, or artificial concatenation boundaries.

## Interpretation and validation

F12 mixes speaker phenotype, phonetic content, microphone response, codec/browser processing and sample-rate effects. No Q-channel regression is performed during extraction. Provenance permits downstream A-only, A+Q, and Q-residualized analyses. Source/library parity, microphone-frequency-response robustness, codec robustness, device robustness, real-data validation, reference validation, and clinical validation remain pending. C067 numerical execution is pending in this local environment; C068 remains research pending source configuration. The proposed 250 ms pause sensitivity work is unrelated to F12.
