"""
Selected tray widget for showing chosen images and quick actions.
"""

from typing import Iterable

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, QMenu
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont, QAction

from ui.components.panels import SelectedImagesPanel


class SelectedTray(QWidget):
    """Right-side tray for selected images with quick actions."""

    selection_changed = Signal(str, bool)  # image_path, is_selected
    clear_requested = Signal()
    review_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)

        title_label = QLabel("Selected")
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #333;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.count_label = QLabel("0 selected")
        self.count_label.setStyleSheet("color: #666; font-size: 12px;")
        header_layout.addWidget(self.count_label)

        layout.addLayout(header_layout)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(6)

        button_style = """
            QPushButton {
                padding: 5px 8px;
                font-size: 11px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f8f8;
            }
            QPushButton:hover:enabled {
                background-color: #e8e8e8;
            }
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
                border-color: #ddd;
            }
        """

        tool_button_style = button_style.replace("QPushButton", "QToolButton") + """
            QToolButton {
                padding-right: 18px;
            }
            QToolButton::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                right: 6px;
                width: 8px;
                height: 8px;
            }
        """

        primary_style = """
            QPushButton {
                padding: 5px 10px;
                font-size: 11px;
                border: 1px solid #007acc;
                border-radius: 4px;
                background-color: #007acc;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #005599;
            }
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
                border-color: #ccc;
            }
        """

        self.export_button = QPushButton("Export")
        self.export_button.setStyleSheet(primary_style)
        self.export_button.clicked.connect(self.export_requested.emit)
        button_layout.addWidget(self.export_button)

        self.actions_button = QToolButton()
        self.actions_button.setStyleSheet(tool_button_style)
        self.actions_button.setPopupMode(QToolButton.InstantPopup)
        self.actions_button.setText("Actions")
        self.actions_menu = QMenu(self.actions_button)

        self.review_action = QAction("Review Selected", self)
        self.review_action.triggered.connect(self.review_requested.emit)
        self.actions_menu.addAction(self.review_action)

        self.clear_action = QAction("Clear Selection", self)
        self.clear_action.triggered.connect(self.clear_requested.emit)
        self.actions_menu.addAction(self.clear_action)

        self.actions_button.setMenu(self.actions_menu)
        button_layout.addWidget(self.actions_button)

        layout.addLayout(button_layout)

        # Selected thumbnails list
        self.selected_panel = SelectedImagesPanel(self, show_header=False)
        self.selected_panel._connect_removal_handler = self._on_remove_clicked
        layout.addWidget(self.selected_panel, 1)

        self._update_controls(0)

    def _on_remove_clicked(self, image_path: str):
        """Emit deselection request for a single image."""
        self.selection_changed.emit(image_path, False)

    def sync_selected_images(self, images: Iterable[str]):
        """Sync the tray with the current selected images."""
        target = set(images)
        current = self.selected_panel.get_selected_images()

        for image_path in current - target:
            self.selected_panel.remove_selected_image(image_path)

        for image_path in sorted(target - current):
            self.selected_panel.add_selected_image(image_path)

        self._update_controls(len(target))

    def clear_selection(self):
        """Clear all selected thumbnails."""
        self.sync_selected_images([])

    def _update_controls(self, count: int):
        self.count_label.setText(f"{count} selected")
        has_selection = count > 0
        self.export_button.setEnabled(has_selection)
        self.actions_button.setEnabled(has_selection)
        self.review_action.setEnabled(has_selection)
        self.clear_action.setEnabled(has_selection)
