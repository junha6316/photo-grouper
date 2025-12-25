"""
Selected photos view component for tab interface.
"""

from typing import List, Set
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QStackedWidget, QButtonGroup
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut

from ui.components.grid_image_view import GridImageView
from ui.components.single_image_view import SingleImageView


class SelectedPhotosView(QWidget):
    """View for displaying selected photos with grid/single view modes."""
    
    selection_changed = Signal(str, bool)  # image_path, is_selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_images: Set[str] = set()
        self.current_view_mode = "grid"  # "grid" or "single"
        self.main_window = parent
        
        self.init_ui()
        self.setup_keyboard_shortcuts()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header with view mode toggles and controls
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: #f8f8f8; border-bottom: 1px solid #ddd;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        
        self.title_label = QLabel("Selected Photos (0 images)")
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #333;")
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # View mode toggle buttons
        view_toggle_layout = QHBoxLayout()
        
        self.grid_view_button = QPushButton("🔲 Grid")
        self.grid_view_button.setCheckable(True)
        self.grid_view_button.setChecked(True)
        self.grid_view_button.clicked.connect(lambda: self.set_view_mode("grid"))
        
        self.single_view_button = QPushButton("🖼️ Single")
        self.single_view_button.setCheckable(True)
        self.single_view_button.clicked.connect(lambda: self.set_view_mode("single"))
        
        # Group buttons for mutual exclusivity
        self.view_button_group = QButtonGroup(self)
        self.view_button_group.addButton(self.grid_view_button)
        self.view_button_group.addButton(self.single_view_button)
        
        # Style view buttons
        view_button_style = """
            QPushButton {
                padding: 6px 12px;
                font-size: 11px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                margin-right: 2px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
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
        
        # Export button
        self.export_button = QPushButton("📁 Export to Folder")
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
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
        """)
        self.export_button.clicked.connect(self.export_selected_images)
        header_layout.addWidget(self.export_button)
        
        # Deselect all button
        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.setEnabled(False)
        self.deselect_all_button.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-size: 11px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QPushButton:hover:enabled {
                background-color: #f0f0f0;
            }
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
            }
        """)
        self.deselect_all_button.clicked.connect(self.deselect_all_images)
        header_layout.addWidget(self.deselect_all_button)
        
        layout.addWidget(header_widget)
        
        # Stacked widget to hold both view modes
        self.stacked_widget = QStackedWidget()
        
        # Default empty state
        self.empty_label = QLabel("No photos selected")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #888; font-size: 16px; padding: 50px;")
        self.stacked_widget.addWidget(self.empty_label)
        
        # Grid view (will be created when images are set)
        self.grid_view = None
        self.grid_view_index = -1
        
        # Single view (will be created when images are set)
        self.single_view = None
        self.single_view_index = -1
        
        layout.addWidget(self.stacked_widget)
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for navigation."""
        # Arrow keys for single view navigation
        left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        left_shortcut.activated.connect(lambda: self.navigate_single_view(-1))
        
        right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        right_shortcut.activated.connect(lambda: self.navigate_single_view(1))
        
        # Space for selection toggle in single view
        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        space_shortcut.activated.connect(self.toggle_current_image_selection)
    
    def update_selected_images(self):
        """Update the view with current selected images from main window."""
        if self.main_window and hasattr(self.main_window, 'global_selected_images'):
            self.selected_images = self.main_window.global_selected_images.copy()
        else:
            self.selected_images = set()
        
        self.update_display()
    
    def update_display(self):
        """Update the display with current selected images."""
        image_list = list(self.selected_images)
        
        # Update title
        self.title_label.setText(f"Selected Photos ({len(image_list)} images)")
        
        # Update buttons
        self.deselect_all_button.setEnabled(len(image_list) > 0)
        self.export_button.setEnabled(len(image_list) > 0)
        
        if not image_list:
            # Show empty state
            self.stacked_widget.setCurrentIndex(0)
            return
        
        # Create or update grid view
        if self.grid_view:
            # Remove old grid view
            self.stacked_widget.removeWidget(self.grid_view)
            self.grid_view.deleteLater()
        
        self.grid_view = GridImageView(image_list, self, thumbnail_size=150)
        self.grid_view.selection_changed.connect(self.on_image_selection_changed)
        # Pre-select all images in grid view
        self.grid_view.set_all_selected(True)
        self.grid_view_index = self.stacked_widget.addWidget(self.grid_view)
        
        # Create or update single view
        if self.single_view:
            # Remove old single view
            self.stacked_widget.removeWidget(self.single_view)
            self.single_view.deleteLater()
        
        self.single_view = SingleImageView(image_list, self)
        self.single_view.selection_changed.connect(self.on_single_view_selection_changed)
        # Pre-select all images in single view
        for image_path in image_list:
            self.single_view.set_selection(image_path, True)
        self.single_view_index = self.stacked_widget.addWidget(self.single_view)
        
        # Set the current view mode
        if self.current_view_mode == "grid" and self.grid_view:
            self.stacked_widget.setCurrentWidget(self.grid_view)
        elif self.current_view_mode == "single" and self.single_view:
            self.stacked_widget.setCurrentWidget(self.single_view)
    
    def set_view_mode(self, mode: str):
        """Switch between grid and single view modes."""
        if mode == self.current_view_mode:
            return
        
        self.current_view_mode = mode
        
        if not self.selected_images:
            return
        
        if mode == "grid":
            if self.grid_view:
                self.stacked_widget.setCurrentWidget(self.grid_view)
                self.grid_view_button.setChecked(True)
                # Sync selections from single view to grid view
                self.sync_selections_to_grid_view()
        elif mode == "single":
            if self.single_view:
                self.stacked_widget.setCurrentWidget(self.single_view)
                self.single_view_button.setChecked(True)
                # Sync selections from grid to single view
                self.sync_selections_to_single_view()
    
    def sync_selections_to_single_view(self):
        """Sync selections from grid view to single view."""
        if self.grid_view and self.single_view:
            selected_images = set(self.grid_view.get_selected_images())
            for image_path in self.selected_images:
                self.single_view.set_selection(image_path, image_path in selected_images)
    
    def sync_selections_to_grid_view(self):
        """Sync selections from single view to grid view."""
        if self.single_view and self.grid_view:
            selected_images = self.single_view.get_selected_images()
            for image_path in self.selected_images:
                should_be_selected = image_path in selected_images
                self.grid_view.set_image_selected(image_path, should_be_selected)
    
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
    
    def on_image_selection_changed(self, image_path: str, is_selected: bool):
        """Handle selection changes from grid view."""
        # Update single view if it exists
        if self.single_view:
            self.single_view.set_selection(image_path, is_selected)
        
        # Update global selection
        self.update_global_selection(image_path, is_selected)
    
    def on_single_view_selection_changed(self, image_path: str, is_selected: bool):
        """Handle selection changes from single view."""
        # Update grid view if it exists
        if self.grid_view:
            self.grid_view.set_image_selected(image_path, is_selected)
        
        # Update global selection
        self.update_global_selection(image_path, is_selected)
    
    def update_global_selection(self, image_path: str, is_selected: bool):
        """Update global selection in main window."""
        # Update local state
        if is_selected:
            self.selected_images.add(image_path)
        else:
            self.selected_images.discard(image_path)
        
        # Update local UI
        self.title_label.setText(f"Selected Photos ({len(self.selected_images)} images)")
        self.deselect_all_button.setEnabled(len(self.selected_images) > 0)
        self.export_button.setEnabled(len(self.selected_images) > 0)
        
        # Emit signal for main window to handle global state
        self.selection_changed.emit(image_path, is_selected)
    
    def deselect_all_images(self):
        """Deselect all images."""
        if self.grid_view:
            self.grid_view.set_all_selected(False)
        
        if self.single_view:
            for image_path in list(self.selected_images):
                self.single_view.set_selection(image_path, False)
        
        # Clear global selection
        if self.main_window and hasattr(self.main_window, 'clear_global_selection'):
            self.main_window.clear_global_selection()
        
        self.selected_images.clear()
        self.update_display()
    
    def export_selected_images(self):
        """Open export dialog for selected images."""
        if not self.selected_images:
            return
        
        from ui.export_dialog import ExportDialog
        dialog = ExportDialog(list(self.selected_images), self)
        dialog.exec()
    
    def sync_selection(self, image_path: str, is_selected: bool):
        """Sync selection state from other views."""
        # Update local selected images set
        if is_selected:
            self.selected_images.add(image_path)
        else:
            self.selected_images.discard(image_path)
        
        # Update displays if the image is in our current view
        if self.grid_view:
            self.grid_view.set_image_selected(image_path, is_selected)
        
        if self.single_view:
            self.single_view.set_selection(image_path, is_selected)
        
        # Update UI
        self.title_label.setText(f"Selected Photos ({len(self.selected_images)} images)")
        self.deselect_all_button.setEnabled(len(self.selected_images) > 0)
        self.export_button.setEnabled(len(self.selected_images) > 0)
