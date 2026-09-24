"""Run-local formant cache keys prevent redundant Burg tracking."""

import pandas as pd

from vslp.acoustic.features.formant_service import FormantService


def test_formant_service_reuses_exact_record_alignment_profile():
    service = FormantService()
    calls = []

    def compute():
        calls.append(1)
        return pd.DataFrame({"f1_token_hz": [300.]}), pd.DataFrame({"time_sec": [.1]})

    first = service.get_or_compute("record", "align1", "profile1", compute)
    second = service.get_or_compute("record", "align1", "profile1", compute)
    assert first is second
    assert len(calls) == 1
    service.get_or_compute("record", "align2", "profile1", compute)
    service.get_or_compute("record", "align1", "profile2", compute)
    service.get_or_compute("other", "align1", "profile1", compute)
    assert len(calls) == 4
    assert service.cached_tracks == 4
