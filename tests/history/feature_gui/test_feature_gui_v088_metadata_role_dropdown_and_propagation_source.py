from pathlib import Path

APP = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")


def _block(name: str) -> str:
    start = APP.index(f"    def {name}")
    end = APP.find("\n    def ", start + 1)
    return APP[start:end if end != -1 else len(APP)]


def test_assigned_role_combo_is_compact_and_table_scoped():
    page = APP[APP.index("    def _metadata_mapping_page"):APP.index("    def update_metadata_mapping_mode_visibility")]
    style = _block("_style_metadata_role_combo")
    refresh = _block("refresh_metadata_mapping_table")
    assert "QListView" in APP
    assert "combo.setFixedHeight(28)" in style
    assert "combo.setMinimumContentsLength(18)" in style
    assert "combo.setMaximumWidth(260)" in style
    assert "combo.view().setMinimumWidth(300)" in style
    assert "QComboBox::drop-down" in style
    assert "self._style_metadata_role_combo(combo)" in refresh
    assert "self.metadata_mapping_table.verticalHeader().setDefaultSectionSize(32)" in page
    assert "self.metadata_mapping_table.setColumnWidth(2, 240)" in page


def test_explicit_metadata_role_mapping_creates_canonical_columns():
    assert "def _accepted_metadata_role_map" in APP
    assert "def _apply_accepted_metadata_roles_to_columns" in APP
    mapping = _block("_accepted_metadata_role_map")
    apply = _block("_apply_accepted_metadata_roles_to_columns")
    standardize = _block("_standardize_metadata_table")
    assert "metadata_mapping_df" in mapping
    assert "canonical_field" in mapping
    assert "self._metadata_role_to_canonical(role)" in mapping
    assert "out[canonical] = values" in apply
    assert "out.loc[empty, canonical]" in apply
    assert "out = self._apply_accepted_metadata_roles_to_columns(out)" in standardize


def test_accept_metadata_mapping_refreshes_downstream_context_controls():
    accept = _block("accept_metadata_mapping")
    assert "self._merge_metadata_context(self.feature_df, self.mapping_df)" in accept
    assert "_refresh_dist_task_combo" in accept
    assert "_refresh_qc_task_combo" in accept
    assert "_refresh_clinical_context_controls" in accept
    assert "_refresh_focus_combos" in accept


def test_context_aliases_include_canonical_metadata_outputs():
    aliases = _block("_context_aliases")
    safe = _block("_safe_context_candidates")
    for token in ["alsfrs_bulbar", "alsfrs_total", "alsbdi_total", "metadata__alsfrs_bulbar", "metadata__alsfrs_total", "metadata__alsbdi_total"]:
        assert token in aliases or token in safe
    for token in ["sex_or_gender", "metadata__sex_or_gender"]:
        assert token in aliases or token in safe
