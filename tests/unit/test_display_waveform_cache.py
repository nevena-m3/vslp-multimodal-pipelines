"""Display decimation never becomes analysis audio or a stale visualization."""

import numpy as np
import pytest
import soundfile as sf

from vslp.gui.acoustic_app.display_waveform import DisplayWaveformCache


def test_display_cache_is_bounded_read_only_and_invalidates(tmp_path):
    path = tmp_path / "synthetic.wav"
    original = np.linspace(-.8, .8, 16000, dtype=np.float32)
    sf.write(path, original, 16000)
    cache = DisplayWaveformCache(capacity=1, points=100)
    times, envelope, duration = cache.get(path)
    assert duration == 1.
    assert len(envelope) <= 200
    assert len(envelope) < len(original)
    assert not times.flags.writeable and not envelope.flags.writeable
    assert cache.get(path)[0] is times
    with pytest.raises(ValueError):
        envelope[0] = 0.
    sf.write(path, np.zeros(32000, dtype=np.float32), 16000)
    updated = cache.get(path)
    assert updated[0] is not times
    assert updated[2] == 2.
