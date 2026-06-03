"""Acoustic feature registry derived from the uploaded Bamboo Passage feature notebook.

The production implementation will compute each feature through plugin classes. This file
locks the first-pass names, subsystems, units, and documentation.
"""

from __future__ import annotations

import pandas as pd


def build_acoustic_feature_registry() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add_feature(name, subsystem, meaning, unit="", computation_note=""):
        rows.append({"feature": name, "subsystem": subsystem, "meaning": meaning, "unit": unit, "computation_note": computation_note})

    for name, meaning, unit in [
        ("speech_rate", "Bamboo words per minute over effective task duration", "words/min"),
        ("total_dur", "Effective analyzed duration after trimming leading/trailing silence", "s"),
        ("speech_dur", "Total non-silent phrase duration inside effective task interval", "s"),
        ("percent_pause", "Internal pause duration divided by effective duration", "proportion"),
        ("num_pause", "Number of internal pauses meeting minimum pause duration", "count"),
        ("mean_pause_dur", "Mean internal pause duration", "s"),
        ("mean_phrase_dur", "Mean speech phrase duration between pauses", "s"),
        ("cv_pause_dur", "Coefficient of variation of internal pause durations", "unitless"),
        ("cv_phrase_dur", "Coefficient of variation of speech phrase durations", "unitless"),
        ("total_pause_dur", "Total internal pause duration", "s"),
    ]:
        add_feature(name, "respiratory_timing", meaning, unit, "RMS/Silero-assisted silence and pause timing.")

    for i in range(1, 6):
        add_feature(f"f{i}", "resonatory", f"Median F{i} across selected high-energy short frames", "Hz", "LPC/formant plugin; verify against reference notebook.")
    for i in range(1, 6):
        add_feature(f"f{i}_bw", "resonatory", f"Median F{i} bandwidth across selected high-energy short frames", "Hz", "LPC root bandwidth.")

    for i in range(1, 4):
        for suffix, meaning in [
            ("d_dx_median", "Median first derivative over selected formant track"),
            ("d_dx_prc_5", "5th percentile of first derivative over selected formant track"),
            ("d_dx_prc_95", "95th percentile of first derivative over selected formant track"),
            ("d_dx_prc_5_95", "95th minus 5th percentile derivative range"),
        ]:
            add_feature(f"f{i}_{suffix}", "articulatory", f"F{i} {meaning}", "Hz/s", "Gradient over selected high-energy frame times.")
    for i in range(1, 4):
        add_feature(f"f{i}_range", "articulatory", f"Robust F{i} range: 95th minus 5th percentile", "Hz", "Computed over selected high-energy frames.")

    add_feature("f0_mean", "phonatory", "Mean voiced fundamental frequency", "Hz", "librosa.yin or validated F0 plugin.")
    add_feature("f0_std", "phonatory", "Standard deviation of voiced fundamental frequency", "Hz", "librosa.yin or validated F0 plugin.")
    add_feature("CPP_mean", "phonatory", "Mean cepstral peak prominence proxy", "dB-like", "Cepstral peak prominence over selected high-energy frames.")
    add_feature("CPP_F1_comp", "coordination", "Complexity index for CPP and F1 tracks", "index", "PCA components explaining 95% variance.")
    add_feature("CPP_F2_comp", "coordination", "Complexity index for CPP and F2 tracks", "index", "PCA components explaining 95% variance.")
    add_feature("F1_F2_comp", "coordination", "Complexity index for F1 and F2 tracks", "index", "PCA components explaining 95% variance.")

    for name, meaning, unit in [
        ("intensity_CV", "Coefficient of variation of RMS intensity", "unitless"),
        ("fft_peaks1", "Strongest envelope modulation frequency", "Hz"),
        ("fft_peaks2", "Second strongest envelope modulation frequency", "Hz"),
        ("fft_ampli1", "Amplitude at strongest envelope modulation peak", "a.u."),
        ("fft_ampli2", "Amplitude at second envelope modulation peak", "a.u."),
        ("nrj_below_boundary", "Envelope modulation energy below 4 Hz", "proportion"),
        ("nrj_above_boundary", "Envelope modulation energy from 4 to 10 Hz", "proportion"),
        ("nrj_3_6", "Envelope modulation energy from 3 to 6 Hz", "proportion"),
        ("ratio_below_above", "Energy below 4 Hz divided by energy 4 to 10 Hz", "ratio"),
    ]:
        add_feature(name, "rhythm", meaning, unit, "Envelope modulation spectrum from filtered amplitude envelope.")

    for name, meaning, unit in [
        ("A1P0", "Amplitude difference between F1-region peak and low-frequency nasal-region peak", "dB"),
        ("A1P0comp", "Binary/continuous compression proxy for A1-P0 relation", "dB"),
        ("A1P1", "Amplitude difference between F1-region peak and ~1 kHz nasal-region peak", "dB"),
        ("A1P1comp", "Binary/continuous compression proxy for A1-P1 relation", "dB"),
        ("A3P0", "Amplitude difference between F3-region peak and low-frequency nasal-region peak", "dB"),
        ("F1freq", "Frequency of F1-region spectral peak", "Hz"),
        ("F1amp", "Amplitude of F1-region spectral peak", "dB"),
        ("F1width", "Approximate F1 bandwidth from LPC summary", "Hz"),
        ("F2freq", "Frequency of F2-region spectral peak", "Hz"),
        ("F2amp", "Amplitude of F2-region spectral peak", "dB"),
        ("F2width", "Approximate F2 bandwidth from LPC summary", "Hz"),
        ("F3freq", "Frequency of F3-region spectral peak", "Hz"),
        ("F3amp", "Amplitude of F3-region spectral peak", "dB"),
        ("F3width", "Approximate F3 bandwidth from LPC summary", "Hz"),
        ("H1freq", "Estimated first harmonic frequency", "Hz"),
        ("H1amp", "Amplitude around first harmonic", "dB"),
        ("H2freq", "Estimated second harmonic frequency", "Hz"),
        ("H2amp", "Amplitude around second harmonic", "dB"),
        ("P0freq", "Low-frequency nasal-region peak frequency", "Hz"),
        ("P0amp", "Low-frequency nasal-region peak amplitude", "dB"),
        ("P0prom", "P0 prominence over local spectral median", "dB"),
        ("P1amp", "~1 kHz nasal-region peak amplitude", "dB"),
        ("RMSamp", "Global RMS amplitude", "linear"),
    ]:
        add_feature(name, "resonatory", meaning, unit, "Welch spectrum plus F0/formant summaries; verify feature validity by task.")

    return pd.DataFrame(rows)


ALL_ACOUSTIC_FEATURES = build_acoustic_feature_registry()["feature"].tolist()
