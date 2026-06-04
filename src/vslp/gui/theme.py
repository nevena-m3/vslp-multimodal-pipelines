"""Shared VSLP Qt theme.

V0.16 branding correction:
- avoids QLabel background styling that produced black artifacts on macOS;
- uses a restrained scientific/clinical dark theme;
- improves readability and scroll behavior on laptop screens.
"""

from __future__ import annotations


def build_dark_stylesheet() -> str:
    return """
    QWidget {
        background-color: #0B1624;
        color: #EEF6FC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        font-size: 13px;
    }
    QMainWindow, QDialog { background-color: #0B1624; }

    QFrame#Sidebar, QFrame#TopBanner, QFrame#BrandingBar, QFrame#Card, QFrame#InfoPanel, QFrame#MetricCard {
        background-color: #122235;
        border: 1px solid #253B52;
        border-radius: 12px;
    }
    QFrame#Sidebar { background-color: #0F2033; }
    QFrame#TopBanner { background-color: #132941; }
    QFrame#BrandingBar { background-color: #0F2033; border-color: #24394E; }
    QFrame#InfoPanel {
        background-color: #102033;
        border: 1px solid #2C4D68;
        border-left: 4px solid #4FA3E3;
    }
    QFrame#MetricCard { background-color: #13283E; }

    QLabel {
        background: transparent;
        color: #EEF6FC;
    }
    QLabel#TitleLabel {
        font-size: 22px;
        font-weight: 800;
        color: #FFFFFF;
    }
    QLabel#AppTitleLabel {
        font-size: 28px;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: 1px;
    }
    QLabel#IPNoticeLabel {
        color: #9CB1C2;
        font-size: 10px;
        line-height: 1.2;
    }
    QLabel#BrandingTitle {
        color: #EAF4FB;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    QLabel#LogoPlaceholder {
        color: #B8CAD8;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid #2F485F;
        border-radius: 8px;
        padding: 6px 10px;
        background-color: #122235;
    }
    QLabel#SubtitleLabel, QLabel#InfoBody {
        color: #B9CBD9;
        font-size: 12px;
    }
    QLabel#SectionHeader, QLabel#InfoTitle {
        color: #F5FAFE;
        font-weight: 800;
        font-size: 14px;
    }
    QLabel#MetricValue {
        color: #FFFFFF;
        font-size: 17px;
        font-weight: 800;
    }
    QLabel#MetricLabel {
        color: #AEC3D4;
        font-size: 11px;
        font-weight: 650;
    }
    QLabel#Chip {
        color: #DDEDF8;
        font-size: 11px;
        font-weight: 700;
        padding: 0px;
        background: transparent;
        border: none;
    }

    QFrame#PlotCanvas {
        background-color: #0A1320;
        border: 1px solid #2C4359;
        border-radius: 12px;
    }
    QLabel#PlotPreviewLabel {
        background-color: #0A1320;
        color: #93A9BB;
        border: 1px dashed #38536B;
        border-radius: 10px;
        padding: 14px;
    }

    QPushButton {
        background-color: #1F5D86;
        color: #FFFFFF;
        border: 1px solid #3D7EA8;
        border-radius: 8px;
        padding: 8px 12px;
        min-height: 22px;
        font-weight: 650;
    }
    QPushButton:hover { background-color: #28719F; }
    QPushButton:pressed { background-color: #174966; }
    QPushButton:disabled {
        background-color: #172737;
        color: #7E91A1;
        border-color: #24384B;
    }
    QPushButton#RunButton {
        background-color: #0E7664;
        border-color: #22A58E;
        font-weight: 800;
    }
    QPushButton#RunButton:hover { background-color: #11927B; }
    QPushButton#OpenButton {
        background-color: #304357;
        border-color: #5E7287;
    }
    QPushButton#DangerButton {
        background-color: #823847;
        border-color: #AA5668;
    }

    QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
        background-color: #0F1D2C;
        color: #F5FAFE;
        border: 1px solid #30495F;
        border-radius: 7px;
        padding: 7px;
        selection-background-color: #2C6F9F;
    }
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #60B6F2;
    }

    QTabWidget::pane {
        border: 1px solid #24394E;
        border-radius: 10px;
        background: #0B1624;
    }
    QTabBar::tab {
        background: #122235;
        color: #C8D8E5;
        border: 1px solid #24394E;
        padding: 8px 12px;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        margin-right: 2px;
        font-weight: 650;
    }
    QTabBar::tab:selected {
        background: #1F5D86;
        color: #FFFFFF;
    }
    QTabBar::tab:hover { background: #1A4464; }

    QGroupBox {
        background-color: #0F1E30;
        border: 1px solid #283E54;
        border-radius: 10px;
        margin-top: 14px;
        padding: 12px;
        color: #F4F9FD;
        font-weight: 750;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        background-color: #0B1624;
    }

    QTreeWidget, QListWidget, QTableWidget {
        background-color: #0E1B2A;
        alternate-background-color: #122335;
        color: #EEF6FC;
        border: 1px solid #283E54;
        border-radius: 8px;
        gridline-color: #30495F;
    }
    QHeaderView::section {
        background-color: #1A344D;
        color: #FFFFFF;
        padding: 6px;
        border: 1px solid #30495F;
        font-weight: 750;
    }
    QTreeWidget::item, QTableWidget::item { padding: 4px; }
    QTreeWidget::item:selected, QTableWidget::item:selected {
        background-color: #236C9E;
        color: #FFFFFF;
    }

    QProgressBar {
        background-color: #0F1D2C;
        border: 1px solid #30495F;
        border-radius: 7px;
        text-align: center;
        min-height: 18px;
        color: #FFFFFF;
        font-weight: 700;
    }
    QProgressBar::chunk {
        background-color: #18A27F;
        border-radius: 6px;
    }

    QCheckBox, QRadioButton { spacing: 7px; background: transparent; }
    QScrollArea { border: none; background: transparent; }
    QSplitter::handle { background-color: #20364B; }
    QToolTip {
        background-color: #14283D;
        color: #FFFFFF;
        border: 1px solid #60B6F2;
        padding: 8px;
        border-radius: 6px;
    }
    QScrollBar:vertical, QScrollBar:horizontal {
        background: #0B1624;
        border: none;
        width: 11px;
        height: 11px;
    }
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
        background: #35566F;
        border-radius: 5px;
        min-height: 28px;
        min-width: 28px;
    }
    QScrollBar::handle:hover { background: #4D7897; }
    QScrollBar::add-line, QScrollBar::sub-line { height: 0px; width: 0px; }
    """
