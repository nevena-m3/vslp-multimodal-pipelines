"""Run-local formant representation shared by aligned formant families."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


class FormantService:
    """Compute one native Burg track per recording, Alignment run, and profile."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}

    def get_or_compute(
        self, recording_id: str, alignment_run_id: str, profile_id: str,
        compute: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = (recording_id, alignment_run_id, profile_id)
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    @property
    def cached_tracks(self) -> int:
        return len(self._cache)
