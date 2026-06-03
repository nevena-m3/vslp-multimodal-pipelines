from vslp.acoustic.metadata.stage import MetadataConfig, _parse_filename_metadata


def test_parse_filename_metadata_basic():
    parsed = _parse_filename_metadata("SUBJ001_SESSION02_ITERATION03_BAMBOO.wav", MetadataConfig(), 1)
    assert parsed["subject_id"] == "001"
    assert parsed["session_id"] == "02"
    assert parsed["iteration"] == "03"
    assert parsed["task"] == "bamboo"
