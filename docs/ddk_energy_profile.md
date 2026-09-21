# DDK Energy profile

Scientific source: Tanchip et al. (2022), *Journal of Speech, Language, and Hearing Research*, DOI [10.1044/2021_JSLHR-21-00503](https://doi.org/10.1044/2021_JSLHR-21-00503), Table 3.

## Paper-defined operations

On a private working copy: remove DC offset, scale by the maximum, zero pad, apply a 200 Hz low-pass FIR filter, calculate sum-of-squares energy in 20 ms frames, and use a 20 ms sliding moving average of the energy as the dynamic threshold. Threshold crossings are retained as raw candidates. The native-rate waveform and source file are unchanged.

## Versioned engineering decisions

`vslp-ddk-tanchip-energy-2` uses a 10 ms hop, a 101-tap linear-phase Hamming FIR (order 100), centered convolution with 100 samples of zero padding on each side, 20 ms minimum event duration, and 10 ms maximum gap for debounce. The paper does not specify these values. Frame completeness padding is separate and never extends final boundaries beyond the source duration. Scaling uses maximum **absolute** amplitude for numerical safety when the largest signed sample is small or negative. This is an explicit deviation from the paper's ambiguous phrase “scaling by maximum value.”

The raw 20 ms threshold can flicker within one sustained energy event. Final events group raw crossings only within one energy-supported island; a 10 ms gap may be bridged. A background support floor is estimated from the median energy and the lower-tail spread (`median + 6 × (median − p10)`). A relative `1e-5 × maximum energy` numerical floor avoids false events created by DC subtraction and filter roundoff during otherwise silent spans. These are implementation rules, not published Tanchip values or clinical thresholds. Raw crossings and raw candidate events remain in output.

DDK rate is the number of final events divided by the interval from first event onset to last event offset. Cycle timing is measured between successive energy peaks. cTV is the mean absolute difference between consecutive cycle durations. After a manual boundary edit or exclusion, the final review layer recalculates timing from reviewed interval midpoints, because an edited event does not have an automatically measured energy nucleus. This timing-source change is explicit in the final decision table.

Weak energy separation, no events, excess candidate fragmentation, and many removed candidates cause review flags. Slow or irregular performance is retained. Automatic review is not a clinical exclusion decision.

## Corpus audit

The opt-in command `python -m vslp.acoustic.segment.ddk_audit INPUT_FOLDER OUTPUT_FOLDER` writes per-file and per-event CSVs plus a JSON summary of sample rates, durations, event counts, cycles, gaps, support, raw crossings, and postprocessing removals. It reads no clinical metadata. No corpus-specific default values are claimed until the user elects to run this command and inspects its results. Increasing the hop or minimum duration can lose short syllables; decreasing either increases candidate fragmentation and computation. Increasing the debounce gap can merge rapid repetitions; decreasing it can split one syllable. FIR order trades transition-band sharpness against computation and edge effects.

Passing synthetic and integration tests establishes software behavior only. It does not establish agreement with manual labels or reproduce the paper's clinical validation statistics.
