from __future__ import annotations

import numpy as np

from vslp.acoustic.preprocess.audio import (
    apply_dc_offset_policy,
    dc_offset_remove,
    resolve_mono_channel,
)


def test_dc_offset_remove_preserves_ac_component():
    sr = 44100
    t = np.arange(sr, dtype=np.float64) / sr
    x = (0.20 * np.sin(2 * np.pi * 220.0 * t) + 0.10).astype(np.float32)

    y = dc_offset_remove(x)

    assert y.dtype == np.float32
    assert abs(float(np.mean(y, dtype=np.float64))) < 1e-7
    expected = x.astype(np.float64) - float(np.mean(x, dtype=np.float64))
    assert np.max(np.abs(y.astype(np.float64) - expected)) < 5e-7


def test_mono_channel_is_selected_directly():
    x = np.linspace(-0.2, 0.2, 1000, dtype=np.float32)[:, None]
    result = resolve_mono_channel(x)
    assert result.status == "resolved_mono"
    assert result.selected_channel == 0


def test_duplicate_stereo_selects_first_channel_deterministically():
    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    ch = (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    x = np.column_stack([ch, ch.copy()])
    result = resolve_mono_channel(x)
    assert result.status == "resolved_duplicate_multichannel"
    assert result.selected_channel == 0


def test_single_usable_channel_is_selected_when_other_channel_is_silent():
    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    ch = (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    x = np.column_stack([np.zeros_like(ch), ch])
    result = resolve_mono_channel(x)
    assert result.status == "resolved_single_usable_channel"
    assert result.selected_channel == 1


def test_non_equivalent_stereo_requires_review():
    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    ch0 = (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    ch1 = (0.2 * np.sin(2 * np.pi * 310.0 * t)).astype(np.float32)
    x = np.column_stack([ch0, ch1])
    result = resolve_mono_channel(x)
    assert result.status == "needs_channel_review"
    assert result.selected_channel is None


def test_gain_mismatched_channels_are_not_silently_treated_as_duplicates():
    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    ch0 = (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    ch1 = (0.25 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    x = np.column_stack([ch0, ch1])
    result = resolve_mono_channel(x)
    assert result.status == "needs_channel_review"
    assert result.selected_channel is None


def test_dc_policy_can_be_disabled_without_changing_waveform():
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    x = (0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.075).astype(np.float32)

    y, before, after = apply_dc_offset_policy(x, remove_dc_offset=False)

    assert np.array_equal(y, x)
    assert before == after
    assert abs(before) > 0.01


def test_dc_policy_enabled_removes_only_constant_offset():
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    x = (0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.075).astype(np.float32)

    y, before, after = apply_dc_offset_policy(x, remove_dc_offset=True)

    expected = x.astype(np.float64) - before
    assert np.max(np.abs(y.astype(np.float64) - expected)) < 5e-7
    assert abs(after) < 1e-7
