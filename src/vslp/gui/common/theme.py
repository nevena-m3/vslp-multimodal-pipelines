"""Shared VSLP Qt dark-theme constants.

The GUI must be professional, low-friction, and clinician/researcher friendly.
"""

VSLP_DARK_QSS = """
QWidget {
    background-color: #071A2D;
    color: #EAF2F8;
    font-family: 'Inter', 'Segoe UI', 'Arial';
    font-size: 13px;
}
QPushButton {
    background-color: #0E3A5B;
    border: 1px solid #2F80ED;
    border-radius: 6px;
    padding: 7px 12px;
}
QPushButton:hover { background-color: #14517A; }
QPushButton:pressed { background-color: #0A2D47; }
QTabWidget::pane { border: 1px solid #224B6D; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
    background-color: #0B253D;
    border: 1px solid #315D7C;
    border-radius: 5px;
    padding: 4px;
}
QGroupBox {
    border: 1px solid #315D7C;
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
"""
