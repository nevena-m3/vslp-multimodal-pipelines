"""F11 and F12 execute independently from final reviewed timing."""

from vslp.acoustic.features.family12_v2 import PUBLIC_IDS
from vslp.acoustic.features.mixed_dispatch import _executors


def test_f11_f12_use_separate_current_executors():
    executors = _executors()
    assert executors["amplitude"][0] == {"F11"}
    assert executors["spectrum"][0] == {"F12"}
    assert executors["amplitude"][1] is not executors["spectrum"][1]
    assert "vowel_mfcc01_mean" in PUBLIC_IDS
    assert "mfcc01_mean" not in PUBLIC_IDS
