"""
Panel components for organizing UI sections.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea
)
from PySide6.QtCore import Qt

from .image_widgets import SelectedThumbnail


class SelectedImagesPanel(QWidget):
    """Panel showing selected images as thumbnails on the right side."""
    
    def __init__(self, parent=None, show_header: bool = True):
        super().__init__(parent)
        self.selected_images = set()
        self.thumbnail_widgets = {}
        self.show_header = show_header
        self.init_ui()
    
    def init_ui(self):
        """Initialize the selected images panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        header_label = QLabel("Selected Images")
        header_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #333; margin-bottom: 5px;"
        )
        
        self.count_label = QLabel("0 selected")
        self.count_label.setStyleSheet(
            "font-size: 12px; color: #666; margin-bottom: 10px;"
        )
        
        if self.show_header:
            layout.addWidget(header_label)
            layout.addWidget(self.count_label)
        
        # Scroll area for thumbnails
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: #f8f8f8;
            }
        """)
        
        # Container widget for thumbnails
        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(5, 5, 5, 5)
        self.container_layout.setSpacing(8)
        self.container_layout.addStretch()  # Push thumbnails to top
        
        self.scroll_area.setWidget(self.container_widget)
        layout.addWidget(self.scroll_area, 1)
    
    def add_selected_image(self, image_path: str):
        """Add an image to the selected images panel."""
        if image_path in self.selected_images:
            return
        
        self.selected_images.add(image_path)
        
        # Create thumbnail widget
        thumbnail = SelectedThumbnail(image_path, size=120)
        
        # Connect to the removal handler if available, otherwise use local removal
        if hasattr(self, '_connect_removal_handler'):
            thumbnail.remove_clicked.connect(self._connect_removal_handler)
        else:
            thumbnail.remove_clicked.connect(self.remove_selected_image)
        
        self.thumbnail_widgets[image_path] = thumbnail
        
        # Insert before the stretch
        index = self.container_layout.count() - 1
        self.container_layout.insertWidget(index, thumbnail)
        
        self.update_count()
    
    def remove_selected_image(self, image_path: str):
        """Remove an image from the selected images panel."""
        if image_path not in self.selected_images:
            return
        
        self.selected_images.remove(image_path)
        
        # Remove thumbnail widget
        if image_path in self.thumbnail_widgets:
            thumbnail = self.thumbnail_widgets[image_path]
            self.container_layout.removeWidget(thumbnail)
            thumbnail.deleteLater()
            del self.thumbnail_widgets[image_path]
        
        self.update_count()
    
    def update_count(self):
        """Update the count label."""
        count = len(self.selected_images)
        self.count_label.setText(f"{count} selected")
    
    def get_selected_images(self):
        """Get set of selected image paths."""
        return self.selected_images.copy()
    
    def clear_selection(self):
        """Clear all selected images."""
        for image_path in list(self.selected_images):
            self.remove_selected_image(image_path)
