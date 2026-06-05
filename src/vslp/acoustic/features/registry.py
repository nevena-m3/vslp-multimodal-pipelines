"""Scientifically annotated acoustic feature registry.

This registry is derived from the user's Bamboo Acoustic Feature Map and ALS Speech
Biomarkers map. It separates feature *definition* from feature *implementation*:
registered features may be implemented, proxy-computed, or intentionally left as NaN
until the formula is validated against reference code/literature.
"""

from __future__ import annotations

import pandas as pd


def build_acoustic_feature_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_feature(
        name: str,
        subsystem: str,
        meaning: str,
        unit: str = "",
        computation_note: str = "",
        formula: str = "",
        task_scope: str = "task_dependent",
        evidence_tier: str = "",
        expected_low: float | None = None,
        expected_high: float | None = None,
        direction_in_als: str = "",
        implementation_status: str = "pending_validation",
    ) -> None:
        rows.append(
            {
                "feature": name,
                "subsystem": subsystem,
                "meaning": meaning,
                "unit": unit,
                "computation_note": computation_note,
                "formula": formula,
                "task_scope": task_scope,
                "evidence_tier": evidence_tier,
                "expected_low": expected_low,
                "expected_high": expected_high,
                "direction_in_als": direction_in_als,
                "implementation_status": implementation_status,
            }
        )

    # Respiratory/timing — segment-derived; strongest validated subset currently.
    timing = [
        ("speech_rate", "Words/min from known task word count over effective task duration", "words/min", "60 * word_count / effective_duration", "passage/sentence when word count is known", "A", 40, 260, "decreases"),
        ("total_dur", "Effective analyzed duration after trimming leading/trailing nonspeech", "s", "end_time - start_time excluding edge nonspeech", "connected speech", "B", 0.5, 1800, "increases with slowing"),
        ("speech_dur", "Total detected speech duration inside effective task interval", "s", "sum(speech segment durations)", "all segmented tasks", "B", 0.1, 1800, "variable"),
        ("percent_pause", "Internal pause duration divided by effective duration", "proportion", "sum(internal pause duration) / effective_duration", "connected speech", "B", 0, 0.85, "increases"),
        ("num_pause", "Number of internal pauses meeting minimum-pause threshold", "count", "count(internal nonspeech segments >= threshold)", "connected speech", "B", 0, 250, "increases"),
        ("mean_pause_dur", "Mean internal pause duration", "s", "mean(internal pause durations)", "connected speech", "B", 0, 10, "increases"),
        ("mean_phrase_dur", "Mean speech phrase duration between pauses", "s", "mean(speech segment durations)", "connected speech", "B", 0, 30, "decreases"),
        ("cv_pause_dur", "Coefficient of variation of internal pause durations", "unitless", "std(pause_duration)/mean(pause_duration)", "connected speech", "C", 0, 5, "increases"),
        ("cv_phrase_dur", "Coefficient of variation of speech phrase durations", "unitless", "std(phrase_duration)/mean(phrase_duration)", "connected speech", "C", 0, 5, "increases"),
        ("total_pause_dur", "Total internal pause duration", "s", "sum(internal pause durations)", "connected speech", "B", 0, 1800, "increases"),
    ]
    for name, meaning, unit, formula, task, tier, lo, hi, direction in timing:
        add_feature(name, "respiratory_timing", meaning, unit, "Computed from Silero speech/nonspeech segments; does not use full-file audio amplitude.", formula, task, tier, lo, hi, direction, "implemented")

    # Formants and articulatory dynamics.
    for i in range(1, 6):
        lo_hi = {1: (150, 1200), 2: (500, 3500), 3: (1200, 4500), 4: (2500, 6000), 5: (3500, 7500)}[i]
        add_feature(f"f{i}", "articulatory", f"Median F{i}; vocal-tract resonance related to articulatory configuration", "Hz", "LPC root tracking; requires validation against reference notebook/Praat-style formants.", "F_k = fs/(2π) angle(z_k)", "vowel/sentence/passage speech regions", "B" if i <= 2 else "C", lo_hi[0], lo_hi[1], "centralization/compression", "pending_validation")
    for i in range(1, 6):
        add_feature(f"f{i}_bw", "articulatory", f"Median F{i} bandwidth; damping/coupling proxy", "Hz", "LPC root bandwidth. F1 bandwidth may also reflect nasality/damping.", "B_k = -fs/π log|z_k|", "vowel/sentence/passage speech regions", "C" if i == 1 else "D", 0, 1000, "may broaden", "pending_validation")
    for i in range(1, 4):
        for suffix, meaning, formula in [
            ("d_dx_median", "median first derivative of formant track", "median(dF/dt)"),
            ("d_dx_prc_5", "5th percentile of formant velocity", "P5(dF/dt)"),
            ("d_dx_prc_95", "95th percentile of formant velocity", "P95(dF/dt)"),
            ("d_dx_prc_5_95", "robust velocity range", "P95(dF/dt)-P5(dF/dt)"),
        ]:
            add_feature(f"f{i}_{suffix}", "articulatory", f"F{i} {meaning}", "Hz/s", "Requires validated formant trajectories and stable frame timing.", formula, "passage/sentence trajectories", "B" if i <= 2 and "5_95" in suffix else "C", None, None, "reduced absolute slope/range", "pending_validation")
    for i in range(1, 4):
        add_feature(f"f{i}_range", "articulatory", f"Robust F{i} range: 95th minus 5th percentile", "Hz", "Computed over speech-region formant tracks.", "P95(F)-P5(F)", "passage/sentence trajectories", "B" if i <= 2 else "C", 0, None, "decreases", "pending_validation")

    # Phonatory.
    add_feature("f0_mean", "phonatory", "Mean voiced fundamental frequency", "Hz", "Current backend uses local autocorrelation proxy; validate against reference/Praat-style F0 before clinical interpretation.", "mean(f0(t))", "vowel/speech voiced regions", "C", 60, 400, "mean is sex/age confounded", "computed_proxy")
    add_feature("f0_std", "phonatory", "Standard deviation of voiced F0", "Hz", "Current backend uses local autocorrelation proxy; semitone scaling is recommended for cross-speaker analysis.", "std(f0(t))", "vowel/speech voiced regions", "B", 0, 120, "decreases with monopitch", "computed_proxy")
    add_feature("CPP_mean", "phonatory", "Mean cepstral peak prominence proxy", "dB-like", "Current backend computes a local CPP proxy. Validate against line-normalized CPP/CPPS reference before clinical interpretation.", "cepstral peak - fitted baseline", "vowel and connected speech", "B", 0, 30, "decreases with dysphonia", "computed_proxy")

    # Rhythm / envelope modulation.
    rhythm = [
        ("intensity_CV", "Coefficient of variation of RMS intensity", "unitless", "std(RMS)/mean(RMS)", 0, 5, "increases with instability"),
        ("fft_peaks1", "Strongest envelope modulation frequency", "Hz", "argmax(|FFT(envelope)|)", 0, 10, "slows/shifts"),
        ("fft_peaks2", "Second strongest envelope modulation frequency", "Hz", "second peak of envelope spectrum", 0, 10, "varies"),
        ("fft_ampli1", "Amplitude at strongest envelope modulation peak", "a.u.", "|FFT(envelope)| at peak 1", 0, None, "decreases with weaker rhythm"),
        ("fft_ampli2", "Amplitude at second envelope modulation peak", "a.u.", "|FFT(envelope)| at peak 2", 0, None, "varies"),
        ("nrj_below_boundary", "Envelope modulation energy below 4 Hz", "proportion", "sum energy 0-4 Hz", 0, 1, "increases"),
        ("nrj_above_boundary", "Envelope modulation energy from 4 to 10 Hz", "proportion", "sum energy 4-10 Hz", 0, 1, "decreases"),
        ("nrj_3_6", "Envelope modulation energy from 3 to 6 Hz", "proportion", "sum energy 3-6 Hz", 0, 1, "varies"),
        ("ratio_below_above", "Energy below 4 Hz divided by energy 4-10 Hz", "ratio", "energy(0-4)/energy(4-10)", 0, 100, "increases"),
    ]
    for name, meaning, unit, formula, lo, hi, direction in rhythm:
        add_feature(name, "rhythm", meaning, unit, "Envelope modulation spectrum from speech-region/effective-task amplitude envelope.", formula, "passage/connected speech", "B" if name in {"nrj_below_boundary", "nrj_above_boundary", "ratio_below_above", "fft_peaks1"} else "C", lo, hi, direction, "implemented")

    # Resonatory/nasality and sentence spectral block.
    reson = [
        ("A1P0", "A1 minus low nasal pole P0 amplitude", "dB", "A1 - P0", "non-nasal sentence/oral passage", "B"),
        ("A1P0comp", "Compensated A1-P0 nasality index", "dB", "A1P0 - vowel compensation", "sentence", "B"),
        ("A1P1", "A1 minus ~1 kHz nasal pole P1 amplitude", "dB", "A1 - P1", "high-vowel sentence", "B"),
        ("A1P1comp", "Compensated A1-P1 nasality index", "dB", "A1P1 - vowel compensation", "sentence", "B"),
        ("A3P0", "A3 minus P0 spectral-tilt nasality index", "dB", "A3 - P0", "sentence", "B"),
        ("P0freq", "Low-frequency nasal pole frequency", "Hz", "peak < 500 Hz", "sentence", "C"),
        ("P0amp", "Low-frequency nasal pole amplitude", "dB", "spectral amplitude at P0", "sentence", "C"),
        ("P0prom", "P0 prominence over local spectral baseline", "dB", "P0 amplitude - local baseline", "sentence", "C"),
        ("P1amp", "~1 kHz nasal pole amplitude", "dB", "spectral amplitude near P1", "sentence", "C"),
    ]
    for name, meaning, unit, formula, task, tier in reson:
        add_feature(name, "resonatory", meaning, unit, "Pending validated nasality implementation; depends strongly on vowel/task and F0/harmonic placement.", formula, task, tier, None, None, "hypernasality generally lowers A1-P0/A1-P1/A3-P0 and increases pole amplitudes", "not_implemented_yet")
    for name in ["F1freq", "F1amp", "F1width", "F2freq", "F2amp", "F2width", "F3freq", "F3amp", "F3width", "H1freq", "H1amp", "H2freq", "H2amp", "RMSamp"]:
        status = "implemented" if name == "RMSamp" else "not_implemented_yet"
        subsystem = "resonatory" if name != "RMSamp" else "rhythm"
        add_feature(name, subsystem, f"Sentence spectral/nasality support feature: {name}", "varies", "Part of sentence/nasality spectral analysis; validate formulas before clinical use.", "see feature map", "sentence", "C", None, None, "task dependent", status)

    # Coordination.
    for name, meaning in [
        ("CPP_F1_comp", "CPP-F1 coordination complexity"),
        ("CPP_F2_comp", "CPP-F2 coordination complexity"),
        ("F1_F2_comp", "F1-F2 articulatory coordination complexity"),
    ]:
        add_feature(name, "coordination", meaning, "index", "Pending validated time-delay cross-correlation/eigenspectrum implementation.", "eigenspectrum of time-delay correlation matrix", "sentence/passage trajectories", "C", None, None, "altered coupling/complexity", "not_implemented_yet")


    return pd.DataFrame(rows)


ALL_ACOUSTIC_FEATURES = build_acoustic_feature_registry()["feature"].tolist()
