"""
All photos view component for displaying all images in a grid layout.
"""

from typing import List, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ui.components.grid_image_view import GridImageView


class AllPhotosView(QWidget):
    """View for displaying all photos in a grid layout."""
    
    selection_changed = Signal(str, bool)  # image_path, is_selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_images: List[str] = []
        self.hidden_count = 0
        self.grid_view: Optional[GridImageView] = None
        self.main_window = parent
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Default message when no images
        self.default_label = QLabel("No images loaded. Select a folder to start.")
        self.default_label.setAlignment(Qt.AlignCenter)
        self.default_label.setStyleSheet("color: #888; font-size: 16px; padding: 50px;")
        layout.addWidget(self.default_label)
    
    def set_images(self, image_paths: List[str], hidden_count: int = 0):
        """Set the images to display."""
        self.all_images = image_paths
        self.hidden_count = hidden_count
        self.update_display()
    
    def update_display(self):
        """Update the display with current images."""
        # Clear existing widgets
        layout = self.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not self.all_images:
            # Show default message
            if self.hidden_count:
                empty_text = f"No in-focus images ({self.hidden_count} hidden out-of-focus)."
            else:
                empty_text = "No images loaded. Select a folder to start."
            self.default_label = QLabel(empty_text)
            self.default_label.setAlignment(Qt.AlignCenter)
            self.default_label.setStyleSheet("color: #888; font-size: 16px; padding: 50px;")
            layout.addWidget(self.default_label)
            return
        
        # Create header
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(10, 10, 10, 5)
        
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(12)
        
        hidden_text = ""
        if self.hidden_count:
            hidden_text = f", {self.hidden_count} hidden out-of-focus"
        self.header_label = QLabel(f"All Photos ({len(self.all_images)} images{hidden_text})")
        self.header_label.setFont(header_font)
        self.header_label.setStyleSheet("color: #333; padding: 5px;")
        header_layout.addWidget(self.header_label)
        
        # Add loading progress label (initially hidden)
        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet("color: #666; padding: 2px; font-size: 11px;")
        self.loading_label.setVisible(False)
        header_layout.addWidget(self.loading_label)
        
        layout.addLayout(header_layout)
        
        # Create grid view
        self.grid_view = GridImageView(self.all_images, self, thumbnail_size=150)
        self.grid_view.selection_changed.connect(self.on_image_selection_changed)
        self.grid_view.batch_progress.connect(self.on_batch_progress)
        
        # Show initial loading state
        self.loading_label.setText(f"Loading images...")
        self.loading_label.setVisible(True)
        
        # Sync with global selection if available
        if self.main_window and hasattr(self.main_window, 'global_selected_images'):
            for image_path in self.main_window.global_selected_images:
                self.grid_view.set_image_selected(image_path, True)
        
        layout.addWidget(self.grid_view)
    
    def on_batch_progress(self, current: int, total: int):
        """Handle batch creation progress."""
        if current < total:
            self.loading_label.setText(f"Loading images... ({current}/{total})")
            self.loading_label.setVisible(True)
        else:
            # All loaded
            self.loading_label.setVisible(False)
    
    def on_image_selection_changed(self, image_path: str, is_selected: bool):
        """Handle image selection changes."""
        # Emit signal for main window to handle
        self.selection_changed.emit(image_path, is_selected)
    
    def sync_selection(self, image_path: str, is_selected: bool):
        """Sync selection state from other views."""
        if self.grid_view:
            self.grid_view.set_image_selected(image_path, is_selected)

    def clear_selection(self):
        """Clear selection state for all images."""
        if self.grid_view:
            self.grid_view.set_all_selected(False)
    
    def clear(self):
        """Clear all images."""
        self.all_images = []
        self.hidden_count = 0
        self.update_display()
