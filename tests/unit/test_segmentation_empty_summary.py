from pathlib import Path
import pandas as pd

from vslp.acoustic.segment.stage import _write_main_segmentation_summary


def test_empty_segmentation_main_summary_has_headers(tmp_path: Path):
    path = tmp_path / "main.csv"
    _write_main_segmentation_summary(pd.DataFrame(), path)
    df = pd.read_csv(path)
    assert list(df.columns)
    assert "file_name" in df.columns
    assert "speech_fraction" in df.columns
    assert len(df) == 0
