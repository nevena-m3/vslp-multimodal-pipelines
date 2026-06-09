from pathlib import Path

from vslp.analysis.kinematics.schemas import LANDMARK_PRESETS, parse_int_list, selected_landmarks_from_preset
from vslp.analysis.kinematics.ingest import discover_videos


def test_landmark_presets_are_unique_and_nonempty():
    assert "ALS oral-motor core 15" in LANDMARK_PRESETS
    for name, vals in LANDMARK_PRESETS.items():
        assert vals
        assert len(vals) == len(set(vals)), name
        assert all(isinstance(v, int) and v >= 0 for v in vals)


def test_parse_int_list_deduplicates_stably():
    assert parse_int_list("13, 14, 13; 61\n291") == (13, 14, 61, 291)


def test_discover_videos_handles_multiple_extensions(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.webm").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("no")
    found = discover_videos(tmp_path, recursive=True, extensions={".mp4", ".webm"})
    assert [p.name for p in found] == ["a.mp4", "b.webm"]
