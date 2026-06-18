from pathlib import Path

APP = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")


def test_v089_has_unique_column_guard_for_metadata_role_propagation():
    assert "def _ensure_unique_columns" in APP
    assert "Setting with non-unique columns is not allowed" in APP
    assert "self._ensure_unique_columns(meta_df.copy(), \"Metadata table\")" in APP
    assert "return self._ensure_unique_columns(out, \"Metadata table\")" in APP


def test_v089_load_and_map_deduplicates_all_loaded_tables_before_mapping():
    assert "self.feature_df = self._ensure_unique_columns(read_table(self.feature_picker.path), \"Primary feature table\")" in APP
    assert "self.qc_df = self._ensure_unique_columns(read_table(self.qc_picker.path), \"QC table\")" in APP
    assert "self.meta_df = self._ensure_unique_columns(read_table(self.meta_picker.path), \"Metadata table\")" in APP
    assert "self.registry_df = self._ensure_unique_columns(read_table(self.registry_picker.path), \"Feature registry\")" in APP


def test_v089_canonical_renames_use_unique_metadata_names_not_repeated_metadata_prefix():
    assert "self._unique_column_name(f\"metadata__{canon}\", used)" in APP
    assert "target = canon if canon not in used else self._unique_column_name(f\"metadata__{canon}\", used)" in APP


def test_v089_filename_column_scoring_uses_noncapturing_extension_regex():
    assert 'r"\\.(?:wav|webm|mp4|avi|mov|csv)$"' in APP
    assert 'r"\\.(wav|webm|mp4|avi|mov|csv)$"' not in APP
