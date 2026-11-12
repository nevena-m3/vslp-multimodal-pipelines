"""Sustained-vowel segmentation (v2): voicing-based episodes and a steady interior.

Phonation is found by voicing rather than by loudness alone:

1. Praat cross-correlation pitch (10 ms) marks voiced frames.
2. A frame counts as phonation when it is voiced and lies within
   ``intensity_range_db`` (25 dB) of the loudest frame. This excludes breaths,
   examiner speech in the background and room noise.
3. Gaps up to ``max_internal_gap_ms`` are bridged and kept as internal voice
   breaks. Episodes shorter than ``min_phonation_ms`` are dropped.

Every episode is reported as a trial, because people often do two or three
/a/ trials. Gaps longer than ``v2_max_internal_gap_ms`` separate trials.
Each trial's stable region is the window of ``stable_duration_ms`` (2 s) with
the steadiest voice: lowest F0 variability in semitones, lowest intensity
variability and fewest unvoiced frames. The onset and offset guards are capped
at 20% and 10% of the trial, so short ALS/PLS trials still get a window. If
less than 2 s is available but at least ``min_stable_ms`` remains, the shorter
core is used and flagged. The analysed trial is the steadiest one.

The method also flags several problems for review:
- silent or empty files
- truncation at the file end
- material that is not a steady vowel, judged by voiced fraction and F0
  variability (for example connected speech uploaded as /a/)
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy import ndimage

ALGORITHM_VERSION = "vslp-phonation-voicing-2"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _tracks(x: np.ndarray, sr: int, config):
    import parselmouth

    sound = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch_cc(time_step=config.hop_ms / 1000,
                              pitch_floor=config.pitch_floor_hz,
                              pitch_ceiling=config.pitch_ceiling_hz)
    hop = config.hop_ms / 1000
    times = np.arange(0, len(x) / sr, hop) + hop / 2
    xs = np.asarray(pitch.xs())
    values = np.asarray(pitch.selected_array["frequency"], dtype=float)
    f0 = np.full(len(times), np.nan)
    if len(xs):
        step = float(pitch.dx)
        index = np.round((times - xs[0]) / step).astype(int)
        inside = (index >= 0) & (index < len(xs))
        f0[inside] = values[index[inside]]
    f0 = np.where(f0 > 0, f0, np.nan)
    window = max(1, round(sr * config.frame_ms / 1000))
    power = ndimage.uniform_filter1d(np.square(np.asarray(x, dtype=np.float64)), size=window,
                                     mode="constant")
    index = np.minimum((times * sr).astype(int), len(x) - 1)
    intensity = 10 * np.log10(np.maximum(power[index], 1e-12))
    return times, f0, intensity


def _stable_window(f0, intensity, a, b, config):
    """Steadiest window inside one episode, after relative onset/offset guards."""
    frames = b - a
    left = a + min(round(config.onset_guard_ms / config.hop_ms), round(0.2 * frames))
    right = b - min(round(config.offset_guard_ms / config.hop_ms), round(0.1 * frames))
    minimum = round(config.min_stable_ms / config.hop_ms)
    if right - left < minimum:
        return None
    length = min(round(config.stable_duration_ms / config.hop_ms), right - left)
    best, best_score = left, np.inf
    for start in range(left, right - length + 1, max(1, round(50 / config.hop_ms))):
        pitch = f0[start:start + length]
        ok = np.isfinite(pitch)
        semitones = 12 * np.log2(pitch[ok] / np.median(pitch[ok])) if ok.sum() > 2 else []
        score = ((float(np.std(semitones)) if len(semitones) else 10.0)
                 + 0.5 * float(np.std(intensity[start:start + length])) + 10 * (1 - ok.mean()))
        if score < best_score:
            best, best_score = start, score
    pitch = f0[best:best + length]
    ok = np.isfinite(pitch)
    return {"start": best, "end": best + length, "score": best_score,
            "f0_median_hz": float(np.median(pitch[ok])) if ok.any() else np.nan,
            "f0_sd_semitones": float(np.std(12 * np.log2(pitch[ok] / np.median(pitch[ok]))))
            if ok.sum() > 2 else np.nan,
            "intensity_sd_db": float(np.std(intensity[best:best + length])),
            "unvoiced_fraction": float(1 - ok.mean())}


def segment_phonation_v2(x: np.ndarray, sr: int, config) -> dict:
    if sr <= 0 or len(x) == 0:
        raise ValueError("Phonation requires non-empty audio and a positive sample rate")
    work = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    hop = config.hop_ms / 1000
    to_sample = lambda frame: int(min(len(x), max(0, round((frame * hop) * sr))))  # noqa: E731
    flags: list[str] = []
    if float(np.max(np.abs(work))) < 1e-3:
        # Digital silence or an empty file (pilot: peak ~6e-5 was taken as phonation).
        times = np.arange(0, len(x) / sr, hop) + hop / 2
        empty = np.full(len(times), np.nan)
        return {"full_samples": None, "stable_samples": None, "internal_breaks_samples": [],
                "phonation_episodes_samples": [], "trials": [], "frame_times_sec": times,
                "activity_rms": np.full(len(times), -120.0), "activity_threshold": empty,
                "f0_hz": empty, "flags": ["silent_or_empty_recording", "no_detectable_phonation"],
                "automatic_status": "REVIEW",
                "processing": {"algorithm": ALGORITHM_VERSION, "config": asdict(config)}}
    times, f0, intensity = _tracks(work, sr, config)
    voiced = np.isfinite(f0)
    loudest = float(np.max(intensity[voiced])) if voiced.any() else float(np.max(intensity))
    active = voiced & (intensity >= loudest - config.intensity_range_db)
    max_gap = round(config.v2_max_internal_gap_ms / config.hop_ms)
    episodes: list[tuple[int, int]] = []
    for a, b in _runs(active):
        if episodes and a - episodes[-1][1] <= max_gap:
            episodes[-1] = (episodes[-1][0], b)
        else:
            episodes.append((a, b))
    min_frames = round(config.min_phonation_ms / config.hop_ms)
    long_enough = [pair for pair in episodes if pair[1] - pair[0] >= min_frames]
    trials = []
    for number, (a, b) in enumerate(long_enough, 1):
        window = _stable_window(f0, intensity, a, b, config)
        trials.append({"trial_index": number, "start_sample": to_sample(a),
                       "end_sample": to_sample(b),
                       "voiced_fraction": float(active[a:b].mean()),
                       "stable": window})
    full = stable = None
    breaks: list[tuple[int, int]] = []
    stability: dict = {}
    if not episodes:
        flags.append("no_detectable_phonation")
    elif not long_enough:
        flags.append("short_phonation_episode")
        a, b = max(episodes, key=lambda pair: pair[1] - pair[0])
        full = (to_sample(a), to_sample(b))
        flags.append("stable_region_unavailable")
    else:
        # Analysed trial: the steadiest one that has a stable window; if none
        # has one, the longest.
        with_window = [trial for trial in trials if trial["stable"]]
        chosen = (min(with_window, key=lambda trial: trial["stable"]["score"]) if with_window
                  else max(trials, key=lambda trial: trial["end_sample"] - trial["start_sample"]))
        a, b = long_enough[chosen["trial_index"] - 1]
        full = (chosen["start_sample"], chosen["end_sample"])
        breaks = [(to_sample(s0 + a), to_sample(e0 + a)) for s0, e0 in _runs(~active[a:b])
                  if e0 - s0 > 1]
        window = chosen["stable"]
        if window is None:
            flags.append("stable_region_unavailable")
        else:
            stable = (to_sample(window["start"]), to_sample(window["end"]))
            if window["end"] - window["start"] < round(config.stable_duration_ms / config.hop_ms):
                flags.append("stable_region_shorter_than_target")
            stability = {f"stable_{key}": value for key, value in window.items()
                         if key not in {"start", "end", "score"}}
            if window["unvoiced_fraction"] > 0.2:
                flags.append("unsteady_voicing_in_stable_region")
            if window["f0_sd_semitones"] > config.max_stable_f0_sd_semitones:
                flags.append("not_a_steady_vowel_check_task")
        if chosen["voiced_fraction"] < 0.6:
            flags.append("not_a_steady_vowel_check_task")
        if len(x) - full[1] < round(0.05 * sr):
            flags.append("phonation_truncated_at_file_end")
    activity_threshold = np.full_like(intensity, loudest - config.intensity_range_db)
    return {
        "full_samples": full, "stable_samples": stable, "internal_breaks_samples": breaks,
        "phonation_episodes_samples": [(to_sample(a), to_sample(b)) for a, b in long_enough],
        "trials": trials, "n_trials": len(trials),
        "frame_times_sec": times, "activity_rms": intensity,
        "activity_threshold": activity_threshold, "f0_hz": f0,
        "flags": sorted(set(flags), key=flags.index),
        "automatic_status": "REVIEW" if flags else "ACCEPTED",
        **stability,
        "processing": {"algorithm": ALGORITHM_VERSION, "voicing": "praat to_pitch_cc",
                       "threshold": "voiced and within intensity_range_db of loudest voiced frame",
                       "trial_choice": "steadiest trial with a stable window",
                       "config": asdict(config)},
    }
