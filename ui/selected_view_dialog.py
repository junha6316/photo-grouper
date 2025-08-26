from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QScrollArea, QPushButton, QWidget, QStackedWidget, QButtonGroup
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from typing import List

from ui.components.grid_image_view import GridImageView
from ui.components.single_image_view import SingleImageView

class SelectedViewDialog(QDialog):
    """Dialog to show all selected photos in a dedicated view with single/grid modes and deselect functionality."""
    
    def __init__(self, selected_images: List[str], parent=None):
        super().__init__(parent)
        self.selected_images = list(selected_images)  # Make a copy
        self.selected_count = len(selected_images)
        self.current_view_mode = "grid"  # "grid" or "single"
        self.main_window = parent  # Reference to main window for global selection sync
        
        self.setWindowTitle(f"Selected Photos - {len(selected_images)} Images")
        self.setGeometry(100, 100, 1000, 700)
        
        self.init_ui()
        self.setup_keyboard_shortcuts()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Header with view mode toggles and selection controls
        header_layout = QHBoxLayout()
        
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        
        title_label = QLabel("Selected Photos")
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
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
        
        # Selection controls
        self.selection_status = QLabel(f"{self.selected_count} selected")
        self.selection_status.setStyleSheet("color: #888; font-size: 12px; margin-right: 10px;")
        header_layout.addWidget(self.selection_status)
        
        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.setStyleSheet("""
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
        self.deselect_all_button.clicked.connect(self.deselect_all_images)
        header_layout.addWidget(self.deselect_all_button)
        
        info_label = QLabel(f"{len(self.selected_images)} images")
        info_label.setStyleSheet("color: #666; font-size: 12px;")
        header_layout.addWidget(info_label)
        
        layout.addLayout(header_layout)
        
        # Stacked widget to hold both view modes
        self.stacked_widget = QStackedWidget()
        
        # Create grid view
        self.grid_view = GridImageView(self.selected_images, self, thumbnail_size=150)
        self.grid_view.selection_changed.connect(self.on_image_selection_changed)
        # Pre-select all images in grid view since these are selected images
        self.grid_view.set_all_selected(True)
        self.stacked_widget.addWidget(self.grid_view)
        
        # Create single image view
        self.single_view = SingleImageView(self.selected_images, self)
        self.single_view.selection_changed.connect(self.on_single_view_selection_changed)
        # Pre-select all images in single view since these are selected images
        for image_path in self.selected_images:
            self.single_view.set_selection(image_path, True)
        self.stacked_widget.addWidget(self.single_view)
        
        layout.addWidget(self.stacked_widget)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close_dialog)
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
    
    
    def set_view_mode(self, mode: str):
        """Switch between grid and single view modes."""
        if mode == self.current_view_mode:
            return
            
        self.current_view_mode = mode
        
        if mode == "grid":
            self.stacked_widget.setCurrentIndex(0)
            self.grid_view_button.setChecked(True)
            # Sync selections from single view to grid view
            self.sync_selections_to_grid_view()
        elif mode == "single":
            self.stacked_widget.setCurrentIndex(1)
            self.single_view_button.setChecked(True)
            # Sync selections from grid to single view
            self.sync_selections_to_single_view()
    
    def sync_selections_to_single_view(self):
        """Sync selections from grid view to single view."""
        image_cards = self.grid_view.get_image_cards()
        for card in image_cards:
            self.single_view.set_selection(card.image_path, card.is_selected)
    
    def sync_selections_to_grid_view(self):
        """Sync selections from single view to grid view."""
        selected_images = self.single_view.get_selected_images()
        for image_path in self.selected_images:
            should_be_selected = image_path in selected_images
            self.grid_view.set_image_selected(image_path, should_be_selected)
    
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
        
        # Tab to switch view modes
        tab_shortcut = QShortcut(QKeySequence(Qt.Key_Tab), self)
        tab_shortcut.activated.connect(self.toggle_view_mode)
        
        # Delete key to deselect current image in single view
        delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        delete_shortcut.activated.connect(self.deselect_current_image)
    
    def navigate_single_view(self, direction: int):
        """Navigate in single view mode (direction: -1 for prev, 1 for next)."""
        if self.current_view_mode == "single":
            if direction < 0:
                self.single_view.prev_image()
            else:
                self.single_view.next_image()
    
    def toggle_current_image_selection(self):
        """Toggle selection of current image in single view mode."""
        if self.current_view_mode == "single":
            self.single_view.toggle_current_selection()
    
    def deselect_current_image(self):
        """Deselect current image in single view mode."""
        if self.current_view_mode == "single":
            current_index = self.single_view.current_index
            if current_index < len(self.selected_images):
                current_image = self.selected_images[current_index]
                self.single_view.set_selection(current_image, False)
                # Also update grid view
                self.grid_view.set_image_selected(current_image, False)
    
    def toggle_view_mode(self):
        """Toggle between grid and single view modes."""
        if self.current_view_mode == "grid":
            self.set_view_mode("single")
        else:
            self.set_view_mode("grid")
    
    def on_single_view_selection_changed(self, image_path: str, is_selected: bool):
        """Handle selection changes from single view."""
        # Update the corresponding grid view
        self.grid_view.set_image_selected(image_path, is_selected)
        
        # Update global selection and UI
        self.update_global_selection(image_path, is_selected)
    
    def on_image_selection_changed(self, image_path: str, is_selected: bool):
        """Handle individual image selection changes from grid view."""
        # Update selection count
        if is_selected:
            self.selected_count += 1
        else:
            self.selected_count -= 1
        
        # Also update single view selection if it exists
        if hasattr(self, 'single_view'):
            self.single_view.set_selection(image_path, is_selected)
        
        # Update global selection and UI
        self.update_global_selection(image_path, is_selected)
    
    def update_global_selection(self, image_path: str, is_selected: bool):
        """Update global selection in main window and UI."""
        # Update global selection in main window
        if self.main_window and hasattr(self.main_window, 'add_to_global_selection'):
            if is_selected:
                self.main_window.add_to_global_selection(image_path)
            else:
                self.main_window.remove_from_global_selection(image_path)
        
        self.update_selection_ui()
    
    def update_selection_ui(self):
        """Update selection UI elements."""
        self.selection_status.setText(f"{self.selected_count} selected")
        self.deselect_all_button.setEnabled(self.selected_count > 0)
    
    def deselect_all_images(self):
        """Deselect all images in the dialog."""
        # Deselect all cards in grid view
        self.grid_view.set_all_selected(False)
        
        # Deselect all in single view
        if hasattr(self, 'single_view'):
            for image_path in self.selected_images:
                self.single_view.set_selection(image_path, False)
        
        print("Deselected all images in selected view")
    
    def close_dialog(self):
        """Close dialog after ensuring all selections are properly synced."""
        # Sync any final selections from single view to grid view
        if self.current_view_mode == "single":
            self.sync_selections_to_grid_view()
        
        # Ensure global selection is up to date with final grid view state
        if self.main_window and hasattr(self.main_window, 'add_to_global_selection'):
            selected_images = self.grid_view.get_selected_images()
            for image_path in self.selected_images:
                if image_path in selected_images:
                    self.main_window.add_to_global_selection(image_path)
                else:
                    self.main_window.remove_from_global_selection(image_path)
        
        self.accept()