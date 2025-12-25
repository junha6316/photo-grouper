"""
Group detail view for reviewing a single cluster in-place.
"""

from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QButtonGroup
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut

from ui.components.grid_image_view import GridImageView
from ui.components.single_image_view import SingleImageView
from core.phash import (
    DEFAULT_MAX_GROUP_SIZE,
    DEFAULT_PHASH_DISTANCE,
    group_paths_by_phash,
)

PHASH_STACK_DISTANCE = DEFAULT_PHASH_DISTANCE
PHASH_STACK_MAX_GROUP_SIZE = DEFAULT_MAX_GROUP_SIZE


class GroupDetailView(QWidget):
    """Embedded view for a single group with selection support."""

    selection_changed = Signal(str, bool)  # image_path, is_selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cluster_images: List[str] = []
        self.cluster_number = 0
        self.is_singles_group = False
        self.similarity: Optional[float] = None
        self.selected_count = 0
        self.current_view_mode = "grid"  # "grid" or "single"
        self.main_window = parent

        self.grid_view = None
        self.single_view = None
        self.grid_view_index = -1
        self.single_view_index = -1

        self.init_ui()
        self.setup_keyboard_shortcuts()
        self._show_empty_state()

    def init_ui(self):
        """Initialize the detail view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header_layout = QHBoxLayout()

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)

        self.title_label = QLabel("Group")
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #333;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        # View mode toggle buttons
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
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
                border-color: #ddd;
            }
        """
        self.grid_view_button.setStyleSheet(view_button_style)
        self.single_view_button.setStyleSheet(view_button_style)

        view_toggle_layout.addWidget(self.grid_view_button)
        view_toggle_layout.addWidget(self.single_view_button)
        header_layout.addLayout(view_toggle_layout)

        header_layout.addStretch()

        # Selection status and controls
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
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
                border-color: #ddd;
            }
        """)
        self.select_all_button.clicked.connect(self.toggle_select_all)
        header_layout.addWidget(self.select_all_button)

        layout.addLayout(header_layout)

        # Content stack
        self.stacked_widget = QStackedWidget()
        self.empty_label = QLabel("Select a group to review")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #888; font-size: 16px; padding: 50px;")
        self.stacked_widget.addWidget(self.empty_label)
        layout.addWidget(self.stacked_widget)

        self._set_controls_enabled(False)

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for navigation."""
        left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        left_shortcut.activated.connect(lambda: self.navigate_single_view(-1))

        right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        right_shortcut.activated.connect(lambda: self.navigate_single_view(1))

        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        space_shortcut.activated.connect(self.toggle_current_image_selection)

        tab_shortcut = QShortcut(QKeySequence(Qt.Key_Tab), self)
        tab_shortcut.activated.connect(self.toggle_view_mode)

    def _set_controls_enabled(self, enabled: bool):
        self.grid_view_button.setEnabled(enabled)
        self.single_view_button.setEnabled(enabled)
        self.select_all_button.setEnabled(enabled)

    def _show_empty_state(self):
        self.stacked_widget.setCurrentWidget(self.empty_label)
        self.title_label.setText("Group")
        self.selection_status.setText("0 selected")
        self.select_all_button.setText("Select All")
        self._set_controls_enabled(False)

    def set_cluster(
        self,
        cluster_images: List[str],
        cluster_number: int,
        is_singles_group: bool = False,
        similarity: Optional[float] = None,
    ):
        """Set the current cluster to display."""
        self.cluster_images = list(cluster_images)
        self.cluster_number = cluster_number
        self.is_singles_group = is_singles_group
        self.similarity = similarity
        self._rebuild_views()

    def update_cluster_images(
        self,
        cluster_images: List[str],
        similarity: Optional[float] = None,
        is_singles_group: Optional[bool] = None,
    ):
        """Update cluster images while keeping view mode and selection."""
        current_mode = self.current_view_mode
        current_image = None
        if current_mode == "single" and self.single_view and self.cluster_images:
            index = self.single_view.current_index
            if 0 <= index < len(self.cluster_images):
                current_image = self.cluster_images[index]

        self.cluster_images = list(cluster_images)
        if similarity is not None:
            self.similarity = similarity
        if is_singles_group is not None:
            self.is_singles_group = is_singles_group
        self._rebuild_views()

        if current_mode == "single" and current_image and self.single_view:
            if current_image in self.cluster_images:
                self.set_view_mode("single")
                self.single_view.go_to_image(self.cluster_images.index(current_image))

    def clear(self):
        """Clear the current cluster view."""
        self.cluster_images = []
        self._rebuild_views()

    def _rebuild_views(self):
        """Rebuild the grid and single views for the current cluster."""
        # Cleanup old views
        if self.grid_view:
            if hasattr(self.grid_view, "cleanup"):
                self.grid_view.cleanup()
            self.stacked_widget.removeWidget(self.grid_view)
            self.grid_view.deleteLater()
            self.grid_view = None

        if self.single_view:
            self.stacked_widget.removeWidget(self.single_view)
            self.single_view.deleteLater()
            self.single_view = None

        if not self.cluster_images:
            self._show_empty_state()
            return

        # Create new views
        stack_groups = group_paths_by_phash(
            self.cluster_images,
            max_distance=PHASH_STACK_DISTANCE,
            max_group_size=PHASH_STACK_MAX_GROUP_SIZE,
        )
        self.grid_view = GridImageView(
            self.cluster_images,
            self,
            thumbnail_size=180,
            stack_groups=stack_groups,
            stack_open_callback=self.open_stack_dialog,
        )
        self.grid_view.selection_changed.connect(self.on_image_selection_changed)
        self.grid_view_index = self.stacked_widget.addWidget(self.grid_view)

        self.single_view = SingleImageView(self.cluster_images, self)
        self.single_view.selection_changed.connect(self.on_single_view_selection_changed)
        self.single_view_index = self.stacked_widget.addWidget(self.single_view)

        self._set_controls_enabled(True)
        self._update_title()
        self.sync_with_global_selection()
        current_mode = self.current_view_mode
        self.current_view_mode = ""
        self.set_view_mode(current_mode)

    def _update_title(self):
        count = len(self.cluster_images)
        if self.is_singles_group:
            title = f"Singles ({count} images)"
        else:
            title = f"Group {self.cluster_number} ({count} images)"

        if self.similarity is not None and self.similarity > 0:
            title += f"  Similarity {self.similarity:.2f}"

        self.title_label.setText(title)

    def set_view_mode(self, mode: str):
        """Switch between grid and single view modes."""
        if mode == self.current_view_mode:
            return

        self.current_view_mode = mode
        if not self.cluster_images:
            return

        if mode == "grid" and self.grid_view:
            self.stacked_widget.setCurrentWidget(self.grid_view)
            self.grid_view_button.setChecked(True)
            self.sync_selections_to_grid_view()
        elif mode == "single" and self.single_view:
            self.stacked_widget.setCurrentWidget(self.single_view)
            self.single_view_button.setChecked(True)
            self.sync_selections_to_single_view()

    def toggle_view_mode(self):
        """Toggle between grid and single view modes."""
        if self.current_view_mode == "grid":
            self.set_view_mode("single")
        else:
            self.set_view_mode("grid")

    def navigate_single_view(self, direction: int):
        """Navigate in single view mode."""
        if self.current_view_mode == "single" and self.single_view:
            if direction < 0:
                self.single_view.prev_image()
            else:
                self.single_view.next_image()

    def toggle_current_image_selection(self):
        """Toggle selection of current image in single view mode."""
        if self.current_view_mode == "single" and self.single_view:
            self.single_view.toggle_current_selection()

    def sync_selections_to_single_view(self):
        """Sync selections from grid view to single view."""
        if not self.grid_view or not self.single_view:
            return
        selected_images = set(self.grid_view.get_selected_images())
        for image_path in self.cluster_images:
            self.single_view.set_selection(image_path, image_path in selected_images)

    def sync_selections_to_grid_view(self):
        """Sync selections from single view to grid view."""
        if not self.single_view or not self.grid_view:
            return
        selected_images = self.single_view.get_selected_images()
        for image_path in self.cluster_images:
            should_be_selected = image_path in selected_images
            self.grid_view.set_image_selected(image_path, should_be_selected)

    def on_image_selection_changed(self, image_path: str, is_selected: bool):
        """Handle selection changes from grid view."""
        if self.single_view:
            self.single_view.set_selection(image_path, is_selected)

        if self.grid_view:
            self.selected_count = len(self.grid_view.get_selected_images())
        else:
            self.selected_count = 0

        self.update_selection_ui()
        self.selection_changed.emit(image_path, is_selected)

    def on_single_view_selection_changed(self, image_path: str, is_selected: bool):
        """Handle selection changes from single view."""
        if self.grid_view:
            self.grid_view.set_image_selected(image_path, is_selected)
            self.selected_count = len(self.grid_view.get_selected_images())
        else:
            self.selected_count = 0

        self.update_selection_ui()
        self.selection_changed.emit(image_path, is_selected)

    def update_selection_ui(self):
        """Update selection UI elements."""
        self.selection_status.setText(f"{self.selected_count} selected")

        if self.selected_count == 0:
            self.select_all_button.setText("Select All")
        elif self.selected_count == len(self.cluster_images):
            self.select_all_button.setText("Deselect All")
        else:
            self.select_all_button.setText("Select All")

    def toggle_select_all(self):
        """Toggle selection of all images in the cluster."""
        if not self.cluster_images or not self.grid_view:
            return

        selected_images = set(self.grid_view.get_selected_images())
        target_state = len(selected_images) != len(self.cluster_images)

        for image_path in self.cluster_images:
            self.grid_view.set_image_selected(image_path, target_state)
            if self.single_view:
                self.single_view.set_selection(image_path, target_state)

        if target_state:
            for image_path in self.cluster_images:
                if image_path not in selected_images:
                    self.selection_changed.emit(image_path, True)
        else:
            for image_path in selected_images:
                self.selection_changed.emit(image_path, False)

        self.selected_count = len(self.cluster_images) if target_state else 0
        self.update_selection_ui()

    def sync_with_global_selection(self):
        """Sync this view with the global selection state."""
        if not self.cluster_images:
            return

        global_selected = set()
        if self.main_window and hasattr(self.main_window, "get_global_selected_images"):
            global_selected = self.main_window.get_global_selected_images()

        self.selected_count = 0

        for image_path in self.cluster_images:
            should_be_selected = image_path in global_selected
            if self.grid_view:
                self.grid_view.set_image_selected(image_path, should_be_selected)
            if self.single_view:
                self.single_view.set_selection(image_path, should_be_selected)
            if should_be_selected:
                self.selected_count += 1

        self.update_selection_ui()

    def sync_selection(self, image_path: str, is_selected: bool):
        """Sync selection state from external changes."""
        if image_path not in self.cluster_images:
            return

        if self.grid_view:
            self.grid_view.set_image_selected(image_path, is_selected)
        if self.single_view:
            self.single_view.set_selection(image_path, is_selected)

        if self.grid_view:
            self.selected_count = len(self.grid_view.get_selected_images())
        else:
            self.selected_count = 0

        self.update_selection_ui()

    def clear_selection(self):
        """Clear selection for the current cluster."""
        if not self.cluster_images:
            return

        if self.grid_view:
            self.grid_view.set_all_selected(False)

        if self.single_view:
            for image_path in list(self.single_view.get_selected_images()):
                self.single_view.set_selection(image_path, False)

        self.selected_count = 0
        self.update_selection_ui()

    def open_stack_dialog(self, stack_images: List[str]):
        """Open a dialog to inspect images inside a stack card."""
        if not stack_images:
            return
        try:
            from ui.stack_detail_dialog import StackDetailDialog
            parent = self.main_window if self.main_window else self
            dialog = StackDetailDialog(stack_images, parent)
            dialog.exec()
        except Exception as exc:
            print(f"Error opening stack detail dialog: {exc}")
            import traceback
            traceback.print_exc()
