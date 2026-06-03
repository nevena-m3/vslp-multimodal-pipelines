"""Shared VSLP Qt theme."""

from __future__ import annotations


def build_dark_stylesheet() -> str:
    return """
    QWidget {
        background-color: #081B2F;
        color: #EAF2F8;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 13px;
    }
    QMainWindow, QDialog {
        background-color: #081B2F;
    }
    QFrame#Sidebar, QFrame#Card, QFrame#TopBanner {
        background-color: #0B253D;
        border: 1px solid #234966;
        border-radius: 12px;
    }
    QLabel#TitleLabel {
        font-size: 22px;
        font-weight: 700;
        color: #F4F8FB;
    }
    QLabel#SubtitleLabel {
        color: #BFD5E6;
        font-size: 12px;
    }
    QLabel#SectionHeader {
        font-size: 16px;
        font-weight: 700;
        color: #F4F8FB;
        padding: 4px 0 8px 0;
    }
    QPushButton {
        background-color: #114A74;
        color: #F6FBFF;
        border: 1px solid #2D6C97;
        border-radius: 10px;
        padding: 10px 14px;
        font-weight: 600;
    }
    QPushButton:hover {
        background-color: #1A5F91;
    }
    QPushButton:pressed {
        background-color: #0D3A5C;
    }
    QPushButton:disabled {
        background-color: #13324D;
        color: #8AA8BE;
        border-color: #1F445F;
    }
    QPushButton#RunButton {
        background-color: #0F6B57;
        border-color: #1A8A70;
    }
    QPushButton#RunButton:hover {
        background-color: #14846B;
    }
    QPushButton#OpenButton {
        background-color: #3C4652;
        border-color: #657587;
    }
    QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
        background-color: #102A43;
        color: #F2F7FB;
        border: 1px solid #315D7C;
        border-radius: 8px;
        padding: 8px;
        selection-background-color: #2D6C97;
    }
    QTabWidget::pane {
        border: 1px solid #234966;
        border-radius: 12px;
        top: -1px;
        background: #081B2F;
    }
    QTabBar::tab {
        background: #0B253D;
        border: 1px solid #234966;
        padding: 10px 16px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 4px;
    }
    QTabBar::tab:selected {
        background: #114A74;
    }
    QGroupBox {
        border: 1px solid #234966;
        border-radius: 10px;
        margin-top: 14px;
        padding-top: 14px;
        font-weight: 700;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
    }
    QTreeWidget, QListWidget {
        background-color: #0B253D;
        border: 1px solid #234966;
        border-radius: 8px;
    }
    QProgressBar {
        background-color: #102A43;
        border: 1px solid #315D7C;
        border-radius: 8px;
        text-align: center;
        min-height: 20px;
    }
    QProgressBar::chunk {
        background-color: #14846B;
        border-radius: 7px;
    }
    QCheckBox, QRadioButton {
        spacing: 8px;
    }
    QScrollArea {
        border: none;
    }
    """
