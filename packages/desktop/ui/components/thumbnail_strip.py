"""
Horizontal thumbnail strip component for image navigation.
"""

from typing import List

from PySide6.QtWidgets import QWidget, QScrollArea, QHBoxLayout
from PySide6.QtCore import Qt, Signal

from .image_widgets import ThumbnailWidget


class ThumbnailStrip(QWidget):
    """Horizontal scrollable strip of thumbnail images for navigation."""
    
    # Signal emitted when a thumbnail is clicked (index)
    thumbnail_clicked = Signal(int)
    
    def __init__(self, images: List[str], height: int = 120, thumbnail_size: int = 120, parent=None):
        """
        Initialize the thumbnail strip.
        
        Args:
            images: List of image paths to display
            height: Fixed height of the strip
            thumbnail_size: Size of each thumbnail
            parent: Parent widget
        """
        super().__init__(parent)
        self.images = images
        self.strip_height = height
        self.thumbnail_size = thumbnail_size
        self.thumbnail_widgets = []
        self.current_index = -1
        
        self.init_ui()
        self.create_thumbnails()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for thumbnails
        self.scroll_area = QScrollArea()
        self.scroll_area.setFixedHeight(self.strip_height)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: #f8f8f8;
            }
        """)
        
        layout.addWidget(self.scroll_area)
    
    def create_thumbnails(self):
        """Create thumbnail widgets for all images."""
        # Widget to hold thumbnails in horizontal layout
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(8)
        
        # Clear existing thumbnails
        self.thumbnail_widgets.clear()
        
        # Create thumbnail widgets
        for i, image_path in enumerate(self.images):
            thumbnail = ThumbnailWidget(image_path, i, size=self.thumbnail_size)
            thumbnail.clicked.connect(self._on_thumbnail_clicked)
            self.thumbnail_widgets.append(thumbnail)
            container_layout.addWidget(thumbnail)
        
        # Add stretch to push thumbnails to the left
        container_layout.addStretch()
        
        self.scroll_area.setWidget(container)
    
    def _on_thumbnail_clicked(self, index: int):
        """Handle internal thumbnail click."""
        if 0 <= index < len(self.images):
            self.set_current_index(index)
            self.thumbnail_clicked.emit(index)
    
    def set_current_index(self, index: int):
        """
        Set the currently selected thumbnail.
        
        Args:
            index: Index of the thumbnail to highlight
        """
        if index == self.current_index:
            return
            
        self.current_index = index
        
        # Update visual highlights
        for i, thumbnail in enumerate(self.thumbnail_widgets):
            thumbnail.set_current(i == index)
        
        # Ensure current thumbnail is visible
        if 0 <= index < len(self.thumbnail_widgets):
            self.ensure_thumbnail_visible(index)
    
    def ensure_thumbnail_visible(self, index: int):
        """
        Ensure the thumbnail at the given index is visible in the scroll area.
        
        Args:
            index: Index of the thumbnail to make visible
        """
        if 0 <= index < len(self.thumbnail_widgets):
            thumbnail = self.thumbnail_widgets[index]
            self.scroll_area.ensureWidgetVisible(thumbnail)
    
    def update_images(self, images: List[str]):
        """
        Update the list of images and recreate thumbnails.
        
        Args:
            images: New list of image paths
        """
        self.images = images
        self.current_index = -1
        self.create_thumbnails()
    
    def get_current_index(self) -> int:
        """Get the currently selected thumbnail index."""
        return self.current_index
    
    def get_thumbnail_count(self) -> int:
        """Get the total number of thumbnails."""
        return len(self.thumbnail_widgets)