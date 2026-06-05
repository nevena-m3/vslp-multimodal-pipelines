"""Feature measurement-scale metadata for VSLP acoustic features.

This module describes the *native measurement scale* of each feature before it is
collapsed to one row per file. It is intentionally separate from the scientific
feature registry: the registry defines what a feature means, while this file
specifies where the feature comes from and how it should be summarized later.
"""

from __future__ import annotations

import pandas as pd


def build_feature_scale_registry(feature_registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return aggregation/measurement-scale metadata for acoustic features.

    The current feature table still stores one value per file, but this registry
    documents the intended native scale and the recommended future reducers. It
    is also used by the aggregation stage to avoid treating every feature as if
    mean/median were equally meaningful.
    """
    reg = feature_registry.copy() if feature_registry is not None else pd.DataFrame()
    features = reg["feature"].astype(str).tolist() if "feature" in reg.columns else []

    def base(feature: str) -> dict[str, object]:
        subsystem = ""
        if not reg.empty and "feature" in reg.columns and feature in set(reg["feature"].astype(str)):
            row = reg.loc[reg["feature"].astype(str).eq(feature)].iloc[0]
            subsystem = str(row.get("subsystem", ""))
        return {
            "feature": feature,
            "subsystem": subsystem,
            "native_scale": "file_scalar",
            "native_source": "one_value_per_file",
            "physiologic_unit": "task/file",
            "current_persistence": "acoustic_features_per_file.csv",
            "recommended_file_reducers": "identity",
            "recommended_group_reducers": "median,iqr,q05,q95,n_nonmissing,missing_fraction",
            "ml_recommendation": "use_with_missingness_and_qc_flags",
            "interpretation_note": "Already collapsed to one value per file in the current implementation.",
        }

    rows: list[dict[str, object]] = []
    timing_segments = {
        "total_dur", "speech_dur", "percent_pause", "num_pause", "mean_pause_dur", "mean_phrase_dur",
        "cv_pause_dur", "cv_phrase_dur", "total_pause_dur", "speech_rate",
    }
    rhythm_frame = {"intensity_CV", "fft_peaks1", "fft_peaks2", "fft_ampli1", "fft_ampli2", "nrj_below_boundary", "nrj_above_boundary", "nrj_3_6", "ratio_below_above"}
    phon_frame = {"f0_mean", "f0_std", "CPP_mean", "HNR", "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter", "ddpJitter", "localShimmer", "localdbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer", "num_voicebreaks", "H1freq", "H1amp", "H2freq", "H2amp"}
    formant_frame = {f"f{i}" for i in range(1, 6)} | {f"f{i}_bw" for i in range(1, 6)} | {f"f{i}_range" for i in range(1, 4)} | {f"f{i}_{suffix}" for i in range(1, 4) for suffix in ["d_dx_median", "d_dx_prc_5", "d_dx_prc_95", "d_dx_prc_5_95"]}
    reson_frame = {"A1P0", "A1P0comp", "A1P1", "A1P1comp", "A3P0", "P0freq", "P0amp", "P0prom", "P1amp", "F1freq", "F1amp", "F1width", "F2freq", "F2amp", "F2width", "F3freq", "F3amp", "F3width", "RMSamp"}
    coord_track = {"CPP_F1_comp", "CPP_F2_comp", "F1_F2_comp"}

    for feature in features:
        row = base(feature)
        if feature in timing_segments:
            row.update({
                "native_scale": "segment_event_or_segment_distribution",
                "native_source": "Silero speech/nonspeech segment table",
                "physiologic_unit": "speech phrase / internal pause / effective task",
                "current_persistence": "per-file scalar plus native segment tables under acoustic/004_features/tables/native_measurements/",
                "recommended_file_reducers": "duration_sum,count,mean,cv,percent_of_effective_duration,word_count_rate_when_available",
                "recommended_group_reducers": "median,iqr,q05,q95,n_files,missing_fraction; avoid simple mean when files/tasks differ strongly",
                "ml_recommendation": "include distribution descriptors and QC flags; preserve task stratification",
                "interpretation_note": "Timing features are physiologically meaningful at phrase/pause-event scale before file-level summarization.",
            })
        elif feature in rhythm_frame:
            row.update({
                "native_scale": "effective_task_envelope_track",
                "native_source": "amplitude-envelope modulation spectrum over first-to-last speech interval",
                "physiologic_unit": "connected-speech rhythm over effective task",
                "current_persistence": "per-file EMS scalar; raw envelope track persistence planned",
                "recommended_file_reducers": "EMS spectral peak and band-energy summaries, not mean over arbitrary frames",
                "recommended_group_reducers": "median,iqr,q05,q95,n_files,missing_fraction; stratify by task",
                "ml_recommendation": "use for passage/connected speech; do not pool with vowel/DDK unless explicitly modeled",
                "interpretation_note": "Internal pauses are part of rhythm; speech-only concatenation is not appropriate for EMS rhythm features.",
            })
        elif feature in phon_frame:
            row.update({
                "native_scale": "voiced_frame_or_period_track",
                "native_source": "voiced frame F0/period/RMS/cepstral trajectories",
                "physiologic_unit": "voiced frame / pseudo-period / sustained phonation",
                "current_persistence": "per-file scalar; track persistence planned for reference validation",
                "recommended_file_reducers": "voiced_median,voiced_iqr,voiced_sd,perturbation_formula,voice_break_count",
                "recommended_group_reducers": "median,iqr,q05,q95,n_files,missing_fraction; report voicing support",
                "ml_recommendation": "prefer sustained vowels for perturbation; use QC/voicing validity flags",
                "interpretation_note": "Jitter/shimmer/HNR depend on voicing and F0 tracking; averaging arbitrary frames is not physiologically meaningful.",
            })
        elif feature in formant_frame:
            row.update({
                "native_scale": "valid_formant_frame_track",
                "native_source": "LPC formant trajectories over valid speech frames",
                "physiologic_unit": "valid formant frame / articulatory trajectory",
                "current_persistence": "per-file scalar; formant-track persistence planned",
                "recommended_file_reducers": "median,robust_range_5_95,trajectory_slope_percentiles,valid_frame_fraction",
                "recommended_group_reducers": "median,iqr,q05,q95,n_files,missing_fraction; report low-validity rates",
                "ml_recommendation": "include validity indicators; avoid interpreting low-validity formant tracks",
                "interpretation_note": "Formant dynamics are trajectory features. The scalar should represent a physiologic trajectory summary, not a blind average.",
            })
        elif feature in reson_frame:
            row.update({
                "native_scale": "valid_spectral_frame_track",
                "native_source": "smoothed spectrum/formant/nasal-pole estimates over valid speech frames",
                "physiologic_unit": "vowel/sentence spectral frame",
                "current_persistence": "per-file scalar; spectral-track persistence planned",
                "recommended_file_reducers": "median spectral contrast,valid_frame_fraction,overlap_warning_fraction",
                "recommended_group_reducers": "median,iqr,q05,q95,n_files,missing_fraction; stratify by task/vowel when possible",
                "ml_recommendation": "treat as task/vowel dependent and QC-sensitive",
                "interpretation_note": "Single-microphone nasality metrics are strongly vowel-, F0-, and spectrum-dependent.",
            })
        elif feature in coord_track:
            row.update({
                "native_scale": "aligned_multitrajectory_window",
                "native_source": "time-aligned CPP/F1/F2 trajectories and lagged correlation matrices",
                "physiologic_unit": "cross-subsystem coupling trajectory",
                "current_persistence": "per-file eigenspectrum scalar; trajectory/correlation matrix persistence planned",
                "recommended_file_reducers": "normalized_participation_ratio plus trajectory validity indicators",
                "recommended_group_reducers": "median,iqr,q05,q95,n_files,missing_fraction; report low-validity rates",
                "ml_recommendation": "use as exploratory coordination feature with validity filtering",
                "interpretation_note": "Coordination complexity is not a monotonic better/worse score; interpret only with task and validity context.",
            })
        rows.append(row)
    return pd.DataFrame(rows)



def build_feature_computation_policy(feature_registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the v0.34 scientific computation policy for acoustic features.

    This table is intentionally user-facing. It answers: for a single audio file,
    what signal/segment support does a feature use and exactly how is the native
    multi-value representation reduced to one file-level scalar?
    """
    reg = feature_registry.copy() if feature_registry is not None else pd.DataFrame()
    features = reg["feature"].astype(str).tolist() if "feature" in reg.columns else []

    timing = {
        "total_dur", "speech_dur", "percent_pause", "num_pause", "mean_pause_dur", "mean_phrase_dur",
        "cv_pause_dur", "cv_phrase_dur", "total_pause_dur", "speech_rate",
    }
    rhythm = {"intensity_CV", "fft_peaks1", "fft_peaks2", "fft_ampli1", "fft_ampli2", "nrj_below_boundary", "nrj_above_boundary", "nrj_3_6", "ratio_below_above"}
    phon = {"f0_mean", "f0_std", "CPP_mean", "HNR", "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter", "ddpJitter", "localShimmer", "localdbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer", "num_voicebreaks", "H1freq", "H1amp", "H2freq", "H2amp"}
    formant = {f"f{i}" for i in range(1, 6)} | {f"f{i}_bw" for i in range(1, 6)} | {f"f{i}_range" for i in range(1, 4)} | {f"f{i}_{suffix}" for i in range(1, 4) for suffix in ["d_dx_median", "d_dx_prc_5", "d_dx_prc_95", "d_dx_prc_5_95"]}
    reson = {"A1P0", "A1P0comp", "A1P1", "A1P1comp", "A3P0", "P0freq", "P0amp", "P0prom", "P1amp", "F1freq", "F1amp", "F1width", "F2freq", "F2amp", "F2width", "F3freq", "F3amp", "F3width", "RMSamp"}
    coord = {"CPP_F1_comp", "CPP_F2_comp", "F1_F2_comp"}

    def family_of(feature: str) -> str:
        if feature in timing: return "Timing / respiratory"
        if feature in rhythm: return "Rhythm / EMS"
        if feature in phon: return "Phonatory"
        if feature in formant: return "Articulatory / formant"
        if feature in reson: return "Resonatory / nasality"
        if feature in coord: return "Coordination"
        return "Other / pending"

    family_policy = {
        "Timing / respiratory": {
            "recommended_tasks": "Connected speech / passage / sentence; not sustained vowel except duration metadata",
            "default_region": "Silero speech + internal pause segment table",
            "native_measurements": "speech segment events; internal pause events; first-speech to last-speech effective task interval",
            "file_level_reduction": "sums, counts, durations, percent of effective task, mean/CV over segment-event distributions",
            "why_this_is_scientific": "Pausing and phrase timing are event-level respiratory/speech-planning phenomena; waveform concatenation would remove the pause physiology.",
            "avoid": "Do not compute from concatenated speech waveform; do not treat leading/trailing silence as internal pause burden.",
        },
        "Rhythm / EMS": {
            "recommended_tasks": "Connected speech / passages / sentence reading; not isolated vowel",
            "default_region": "effective_task",
            "native_measurements": "amplitude envelope over first speech onset to last speech offset with internal pauses preserved; 0-10 Hz modulation spectrum",
            "file_level_reduction": "dominant modulation frequencies, peak amplitudes, normalized band powers, slow/fast band ratio, RMS intensity CV",
            "why_this_is_scientific": "Connected-speech rhythm depends on syllabic amplitude modulation plus pause timing; speech-only concatenation distorts low-frequency rhythm.",
            "avoid": "Do not concatenate speech segments for EMS rhythm; do not pool across tasks with very different elicitation designs without task labels.",
        },
        "Phonatory": {
            "recommended_tasks": "Sustained vowel for perturbation; connected speech acceptable for F0/CPP/HNR summaries with caution",
            "default_region": "speech_only with voiced-frame filtering",
            "native_measurements": "voiced F0/period track, RMS-amplitude track, cepstral/HNR frame support, internal voicing gaps",
            "file_level_reduction": "F0 mean/SD across voiced support; perturbation formulas over period/amplitude tracks; CPP/HNR robust summary; voice-break count",
            "why_this_is_scientific": "Phonation features describe vocal-fold vibration and voiced support; silence and unvoiced consonants should not dominate the scalar value.",
            "avoid": "Do not compute jitter/shimmer over unvoiced or low-voicing support; do not interpret without F0/voicing QC.",
        },
        "Articulatory / formant": {
            "recommended_tasks": "Vowels, sentences, passage speech; DDK only if formant tracking is task-justified",
            "default_region": "valid speech frames",
            "native_measurements": "LPC-derived F1-F5 and bandwidth trajectories with valid-frame filters",
            "file_level_reduction": "median formants/bandwidths, 5-95% robust ranges, derivative/slope medians and 5th/95th percentiles",
            "why_this_is_scientific": "Articulatory features are trajectory summaries of tongue/jaw/lip-related vocal-tract resonances; robust medians/ranges reduce bad-frame influence.",
            "avoid": "Do not use blind means over invalid LPC roots; do not ignore valid-frame fraction.",
        },
        "Resonatory / nasality": {
            "recommended_tasks": "Controlled sentences / oral-nasal contrasts / vowel-targeted frames; task and vowel context are critical",
            "default_region": "valid spectral speech frames",
            "native_measurements": "smoothed spectral-frame estimates of A1/A3, P0/P1, F1-F3 support, bandwidths, RMS",
            "file_level_reduction": "median spectral contrasts and support variables with P0-F1 overlap and valid-frame warnings",
            "why_this_is_scientific": "Single-microphone nasality is spectral-frame and vowel dependent; robust spectral contrasts are safer than full-file averages.",
            "avoid": "Do not interpret as nasometer-equivalent; do not ignore vowel/F0 harmonic placement effects.",
        },
        "Coordination": {
            "recommended_tasks": "Connected speech / sentence / passage with enough valid CPP and formant trajectory support",
            "default_region": "effective_task aligned trajectories",
            "native_measurements": "time-aligned CPP, F1, and F2 trajectories; lagged correlation matrices over short delays",
            "file_level_reduction": "normalized participation-ratio eigenspectrum summary for each pairwise trajectory coupling",
            "why_this_is_scientific": "Coordination features summarize coupling structure across speech subsystems, not the average value of either trajectory.",
            "avoid": "Do not interpret as a monotonic severity score; do not compute when trajectory support is sparse or rank-deficient.",
        },
        "Other / pending": {
            "recommended_tasks": "Depends on final feature definition",
            "default_region": "not finalized",
            "native_measurements": "not finalized",
            "file_level_reduction": "not finalized",
            "why_this_is_scientific": "Feature remains registered but needs explicit native-scale policy before interpretation.",
            "avoid": "Do not use for ML until policy and implementation are validated.",
        },
    }

    rows = []
    for feature in features:
        fam = family_of(feature)
        policy = family_policy[fam]
        reg_row = pd.Series(dtype=object)
        if not reg.empty and "feature" in reg.columns and feature in set(reg["feature"].astype(str)):
            reg_row = reg.loc[reg["feature"].astype(str).eq(feature)].iloc[0]
        rows.append({
            "feature": feature,
            "subsystem": reg_row.get("subsystem", fam),
            "family_policy": fam,
            "task_scope": reg_row.get("task_scope", ""),
            "unit": reg_row.get("unit", ""),
            **policy,
        })
    return pd.DataFrame(rows)


def build_feature_family_policy_summary() -> pd.DataFrame:
    """Compact GUI-facing policy summary by feature family."""
    rows = [
        {
            "family": "Timing / respiratory",
            "best_tasks": "Bamboo, passage, reading, free speech",
            "native_scale": "speech/pause events",
            "default_region": "segment table",
            "scalar_reduction": "sum, count, percent, mean/CV of events",
            "core_rule": "Never concatenate speech for pause physiology.",
        },
        {
            "family": "Rhythm / EMS",
            "best_tasks": "Connected speech",
            "native_scale": "effective-task envelope",
            "default_region": "first speech → last speech, pauses preserved",
            "scalar_reduction": "modulation peaks and normalized band powers",
            "core_rule": "Internal pauses are part of rhythm.",
        },
        {
            "family": "Phonatory",
            "best_tasks": "Sustained vowel; voiced connected speech with caution",
            "native_scale": "voiced frames / period support",
            "default_region": "speech + voiced-frame filter",
            "scalar_reduction": "voiced mean/SD, perturbation formulas, CPP/HNR summaries",
            "core_rule": "Silence and unvoiced support should not drive voice features.",
        },
        {
            "family": "Articulatory / formant",
            "best_tasks": "Vowels, sentences, passage",
            "native_scale": "valid LPC formant trajectories",
            "default_region": "valid speech frames",
            "scalar_reduction": "medians, 5-95 ranges, slope percentiles",
            "core_rule": "Use robust trajectory summaries and validity flags.",
        },
        {
            "family": "Resonatory / nasality",
            "best_tasks": "Controlled sentences / vowel-targeted frames",
            "native_scale": "valid spectral frames",
            "default_region": "valid spectral speech frames",
            "scalar_reduction": "median A1/P0/P1/A3 contrasts + warnings",
            "core_rule": "Treat as vowel-, F0-, and device-sensitive.",
        },
        {
            "family": "Coordination",
            "best_tasks": "Connected speech with stable CPP/F1/F2 tracks",
            "native_scale": "aligned multitrajectory windows",
            "default_region": "effective-task trajectories",
            "scalar_reduction": "lagged correlation eigenspectrum complexity",
            "core_rule": "Not a monotonic severity score; require valid support.",
        },
    ]
    return pd.DataFrame(rows)
