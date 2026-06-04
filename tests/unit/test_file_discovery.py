from pathlib import Path

from vslp.core.io import discover_files_with_duplicate_report


def test_same_stem_duplicate_priority_prefers_wav_then_mp4_then_webm(tmp_path: Path):
    wav = tmp_path / "SUBJ001_BAMBOO.wav"
    mp4 = tmp_path / "SUBJ001_BAMBOO.mp4"
    webm = tmp_path / "SUBJ001_BAMBOO.webm"
    other = tmp_path / "SUBJ002_BAMBOO.webm"
    for path in [webm, mp4, wav, other]:
        path.write_bytes(b"x")

    files, skipped = discover_files_with_duplicate_report(
        tmp_path,
        [".wav", ".mp4", ".webm"],
    )

    assert wav in files
    assert other in files
    assert mp4 not in files
    assert webm not in files
    assert len(skipped) == 2
    assert {row["skipped_file_name"] for row in skipped} == {"SUBJ001_BAMBOO.mp4", "SUBJ001_BAMBOO.webm"}


def test_same_stem_duplicate_priority_prefers_mp4_when_wav_absent(tmp_path: Path):
    mp4 = tmp_path / "SUBJ001_DDK.mp4"
    webm = tmp_path / "SUBJ001_DDK.webm"
    for path in [webm, mp4]:
        path.write_bytes(b"x")

    files, skipped = discover_files_with_duplicate_report(tmp_path, [".mp4", ".webm"])

    assert files == [mp4]
    assert len(skipped) == 1
    assert skipped[0]["skipped_file_name"] == "SUBJ001_DDK.webm"
