import os
from PySide6.QtWidgets import (
    QLabel, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor

from ui.utils.image_utils import load_image_as_pixmap



class ZoomableImageLabel(QLabel):
    """Image label with click-to-zoom functionality."""
    
    clicked = Signal()
    
    def __init__(self, max_size: int = 500):
        super().__init__()
        self.max_size = max_size
        self.image_path = ""
        self.is_zoomed = False
        self.original_pixmap = None
        self.zoomed_pixmap = None
        
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(max_size, max_size)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: #fafafa;
                padding: 10px;
            }
            QLabel:hover {
                border-color: #007acc;
                background-color: #f0f8ff;
            }
        """)
        
        # Set cursor to indicate clickability
        self.setCursor(QCursor(Qt.PointingHandCursor))
    
    def set_image(self, image_path: str):
        """Load and display an image."""
        self.image_path = image_path
        self.is_zoomed = False
        self.setText("📷 Loading...")
        
        # Load image directly for better control
        self._load_image_directly()
    
    def _load_image_directly(self):
        """Load image directly using centralized image loading utility."""
        # Load normal size version
        self.original_pixmap = load_image_as_pixmap(
            self.image_path, max_size=self.max_size
        )
        
        # Load zoomed version (2x larger)
        self.zoomed_pixmap = load_image_as_pixmap(
            self.image_path, max_size=self.max_size * 2
        )
        
        if self.original_pixmap and self.zoomed_pixmap:
            # Display normal size initially
            self.setPixmap(self.original_pixmap)
            
            # Update tooltip
            filename = os.path.basename(self.image_path)
            self.setToolTip(f"{filename}\nClick to zoom in/out\n{self.image_path}")
        else:
            # Show error
            filename = os.path.basename(self.image_path)
            self.setText(f"⚠️ Error loading:\n{filename}")
            self.setStyleSheet(self.styleSheet() + """
                QLabel { 
                    color: red; 
                    font-size: 12px;
                }
            """)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for zooming."""
        if event.button() == Qt.LeftButton:
            self.toggle_zoom()
            self.clicked.emit()
        super().mousePressEvent(event)
    
    def toggle_zoom(self):
        """Toggle between normal and zoomed view."""
        if not self.original_pixmap or not self.zoomed_pixmap:
            return
        
        self.is_zoomed = not self.is_zoomed
        
        if self.is_zoomed:
            self.setPixmap(self.zoomed_pixmap)
            self.setToolTip(self.toolTip().replace("Click to zoom in", "Click to zoom out"))
        else:
            self.setPixmap(self.original_pixmap)
            self.setToolTip(self.toolTip().replace("Click to zoom out", "Click to zoom in"))
