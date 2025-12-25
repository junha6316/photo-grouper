from typing import List

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QButtonGroup,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut

from ui.components.grid_image_view import GridImageView
from ui.components.single_image_view import SingleImageView


class StackDetailDialog(QDialog):
    """Dialog to inspect images inside a stack card."""

    def __init__(self, stack_images: List[str], parent=None):
        super().__init__(parent)
        self.stack_images = list(stack_images)
        self.selected_count = 0
        self.current_view_mode = "grid"
        self.main_window = parent

        self.setWindowTitle(f"Stack - {len(self.stack_images)} Images")
        self.setGeometry(120, 120, 1000, 700)

        self.init_ui()
        self.setup_keyboard_shortcuts()
        self.sync_with_global_selection()

    def init_ui(self):
        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)

        title_label = QLabel("Stack")
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        view_toggle_layout = QHBoxLayout()

        self.grid_view_button = QPushButton("Grid")
        self.grid_view_button.setCheckable(True)
        self.grid_view_button.setChecked(True)
        self.grid_view_button.clicked.connect(lambda: self.set_view_mode("grid"))

        self.single_view_button = QPushButton("Single")
        self.single_view_button.setCheckable(True)
        self.single_view_button.clicked.connect(lambda: self.set_view_mode("single"))

        self.view_button_group = QButtonGroup(self)
        self.view_button_group.addButton(self.grid_view_button)
        self.view_button_group.addButton(self.single_view_button)

        view_button_style = """
            QPushButton {
                padding: 6px 12px;
                font-size: 11px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f8f8;
                margin-right: 2px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
            }
            QPushButton:checked {
                background-color: #007acc;
                color: white;
                border-color: #005599;
            }
        """
        self.grid_view_button.setStyleSheet(view_button_style)
        self.single_view_button.setStyleSheet(view_button_style)

        view_toggle_layout.addWidget(self.grid_view_button)
        view_toggle_layout.addWidget(self.single_view_button)
        header_layout.addLayout(view_toggle_layout)

        header_layout.addStretch()

        self.selection_status = QLabel("0 selected")
        self.selection_status.setStyleSheet(
            "color: #888; font-size: 12px; margin-right: 10px;"
        )
        header_layout.addWidget(self.selection_status)

        self.select_all_button = QPushButton("Select All")
        self.select_all_button.setStyleSheet("""
            QPushButton {
                padding: 5px 10px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
                margin-right: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.select_all_button.clicked.connect(self.toggle_select_all)
        header_layout.addWidget(self.select_all_button)

        info_label = QLabel(f"{len(self.stack_images)} images")
        info_label.setStyleSheet("color: #666; font-size: 12px;")
        header_layout.addWidget(info_label)

        layout.addLayout(header_layout)

        self.stacked_widget = QStackedWidget()

        self.grid_view = GridImageView(self.stack_images, self, thumbnail_size=150)
        self.grid_view.selection_changed.connect(self.on_image_selection_changed)
        self.stacked_widget.addWidget(self.grid_view)

        self.single_view = SingleImageView(self.stack_images, self)
        self.single_view.selection_changed.connect(self.on_single_view_selection_changed)
        self.stacked_widget.addWidget(self.single_view)

        layout.addWidget(self.stacked_widget)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

    def setup_keyboard_shortcuts(self):
        left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        left_shortcut.activated.connect(lambda: self.navigate_single_view(-1))

        right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        right_shortcut.activated.connect(lambda: self.navigate_single_view(1))

        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        space_shortcut.activated.connect(self.toggle_current_image_selection)

        tab_shortcut = QShortcut(QKeySequence(Qt.Key_Tab), self)
        tab_shortcut.activated.connect(self.toggle_view_mode)

    def set_view_mode(self, mode: str):
        if mode == self.current_view_mode:
            return
        self.current_view_mode = mode
        if mode == "grid":
            self.stacked_widget.setCurrentWidget(self.grid_view)
            self.grid_view_button.setChecked(True)
            self.sync_selections_to_grid_view()
        elif mode == "single":
            self.stacked_widget.setCurrentWidget(self.single_view)
            self.single_view_button.setChecked(True)
            self.sync_selections_to_single_view()

    def toggle_view_mode(self):
        if self.current_view_mode == "grid":
            self.set_view_mode("single")
        else:
            self.set_view_mode("grid")

    def navigate_single_view(self, direction: int):
        if self.current_view_mode == "single":
            if direction < 0:
                self.single_view.prev_image()
            else:
                self.single_view.next_image()

    def toggle_current_image_selection(self):
        if self.current_view_mode == "single":
            self.single_view.toggle_current_selection()

    def sync_selections_to_single_view(self):
        selected_images = set(self.grid_view.get_selected_images())
        for image_path in self.stack_images:
            self.single_view.set_selection(image_path, image_path in selected_images)

    def sync_selections_to_grid_view(self):
        selected_images = self.single_view.get_selected_images()
        for image_path in self.stack_images:
            self.grid_view.set_image_selected(image_path, image_path in selected_images)

    def sync_with_global_selection(self):
        if not self.stack_images:
            return
        global_selected = set()
        if self.main_window and hasattr(self.main_window, "get_global_selected_images"):
            global_selected = self.main_window.get_global_selected_images()
        for image_path in self.stack_images:
            should_be_selected = image_path in global_selected
            self.grid_view.set_image_selected(image_path, should_be_selected)
            self.single_view.set_selection(image_path, should_be_selected)
        self._update_selected_count()

    def on_image_selection_changed(self, image_path: str, is_selected: bool):
        if self.single_view:
            self.single_view.set_selection(image_path, is_selected)
        self._update_global_selection(image_path, is_selected)
        self._update_selected_count()

    def on_single_view_selection_changed(self, image_path: str, is_selected: bool):
        if self.grid_view:
            self.grid_view.set_image_selected(image_path, is_selected)
        self._update_global_selection(image_path, is_selected)
        self._update_selected_count()

    def _update_global_selection(self, image_path: str, is_selected: bool):
        if self.main_window and hasattr(self.main_window, "add_to_global_selection"):
            if is_selected:
                self.main_window.add_to_global_selection(image_path)
            else:
                self.main_window.remove_from_global_selection(image_path)

    def _update_selected_count(self):
        if self.grid_view:
            self.selected_count = len(self.grid_view.get_selected_images())
        else:
            self.selected_count = 0
        self.update_selection_ui()

    def update_selection_ui(self):
        self.selection_status.setText(f"{self.selected_count} selected")
        if self.selected_count == 0:
            self.select_all_button.setText("Select All")
        elif self.selected_count == len(self.stack_images):
            self.select_all_button.setText("Deselect All")
        else:
            self.select_all_button.setText("Select All")

    def toggle_select_all(self):
        if not self.stack_images or not self.grid_view:
            return

        selected_images = set(self.grid_view.get_selected_images())
        target_state = len(selected_images) != len(self.stack_images)

        self.grid_view.set_all_selected(target_state)
        if self.single_view:
            for image_path in self.stack_images:
                self.single_view.set_selection(image_path, target_state)

        if target_state:
            for image_path in self.stack_images:
                if image_path not in selected_images:
                    self._update_global_selection(image_path, True)
        else:
            for image_path in selected_images:
                self._update_global_selection(image_path, False)

        self.selected_count = len(self.stack_images) if target_state else 0
        self.update_selection_ui()
