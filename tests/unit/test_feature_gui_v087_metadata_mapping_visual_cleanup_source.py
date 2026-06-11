from pathlib import Path

APP = Path("src/vslp/gui/features/app.py").read_text(encoding="utf-8")


def _metadata_mapping_block() -> str:
    start = APP.index("    def _metadata_mapping_page")
    end = APP.index("    def update_metadata_mapping_mode_visibility", start)
    return APP[start:end]


def test_metadata_mapping_page_is_top_aligned_and_not_vertical_stretched():
    block = _metadata_mapping_block()
    assert "layout.setContentsMargins(10, 8, 10, 10)" in block
    assert "layout.setAlignment(Qt.AlignTop)" in block
    assert "card.layout.setAlignment(Qt.AlignTop)" in block
    assert "self.filename_metadata_fallback_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)" in block
    assert "fallback_layout.setAlignment(Qt.AlignTop)" in block
    assert "layout.addWidget(card, 0, Qt.AlignTop)" in block
    assert "layout.addStretch(1)" in block


def test_filename_fallback_label_styles_are_transparent_not_boxed():
    block = _metadata_mapping_block()
    assert 'self.filename_metadata_fallback_frame.setObjectName("FilenameFallbackFrame")' in block
    assert "QFrame#FilenameFallbackFrame QLabel" in block
    assert "background:transparent" in block
    assert "border:none" in block
    assert 'context_source_label = QLabel("Context source")' in block
    assert 'filename_column_label = QLabel("Filename column")' in block
    assert 'context_source_label.setStyleSheet(f"color:{MUTED}; font-size:11px; font-weight:800; background:transparent; border:none; padding:0px;")' in block
    assert 'filename_column_label.setStyleSheet(f"color:{MUTED}; font-size:11px; font-weight:800; background:transparent; border:none; padding:0px;")' in block


def test_filename_fallback_card_frame_scope_does_not_style_all_qframes():
    block = _metadata_mapping_block()
    assert "QFrame#FilenameFallbackFrame" in block
    assert "QFrame {{ background:#F8FBFE" not in block
