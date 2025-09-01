"""
Grouped photos view component - wrapper for the existing PreviewPanel.
"""

from typing import List, Dict
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel, QStackedWidget
from PySide6.QtCore import Signal, Qt

from ui.preview_panel import PreviewPanel


class GroupedPhotosView(QWidget):
    """View for displaying grouped photos using the existing PreviewPanel."""
    
    selection_changed = Signal(str, bool)  # image_path, is_selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_groups: List[List[str]] = []
        self.main_window = parent
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Use QStackedWidget to properly switch between views
        self.stacked_widget = QStackedWidget()
        
        # Create preview panel with scroll area
        self.preview_panel = PreviewPanel()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.preview_panel)
        self.scroll_area.setWidgetResizable(True)
        
        # Create processing message label
        self.processing_label = QLabel()
        self.processing_label.setAlignment(Qt.AlignCenter)
        self.processing_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 16px;
                padding: 50px;
                background-color: #f8f8f8;
                min-height: 200px;
            }
        """)
        self.processing_label.setText("No groups to display")
        
        # Add both to stacked widget
        self.stacked_widget.addWidget(self.scroll_area)  # Index 0
        self.stacked_widget.addWidget(self.processing_label)  # Index 1
        
        # Default to showing scroll area
        self.stacked_widget.setCurrentIndex(0)
        
        layout.addWidget(self.stacked_widget)
    
    def set_groups(self, groups: List[List[str]], min_display_size: int = 2, similarities: List[float] = None):
        """Set the photo groups to display."""
        self.current_groups = groups
        # Switch to preview panel
        self.stacked_widget.setCurrentIndex(0)
        self.preview_panel.display_groups(groups, min_display_size, similarities)
    
    def show_processing_message(self, message: str):
        """Show a processing message instead of the preview panel."""
        self.processing_label.setText(f"⏳ {message}")
        # Switch to processing label
        self.stacked_widget.setCurrentIndex(1)
    
    def update_processing_progress(self, progress: int, message: str):
        """Update the processing message with progress."""
        if self.stacked_widget.currentIndex() == 1:  # Processing label is visible
            self.processing_label.setText(f"⏳ {message}\n\nProgress: {progress}%")
    
    def clear(self):
        """Clear the display."""
        self.current_groups = []
        self.preview_panel.clear()
        self.processing_label.setText("No groups to display")
        # Switch back to scroll area (empty state)
        self.stacked_widget.setCurrentIndex(0)
    
    def get_preview_panel(self) -> PreviewPanel:
        """Get the underlying preview panel."""
        return self.preview_panel
    
    def sync_selection(self, image_path: str, is_selected: bool):
        """Sync selection state from other views."""
        # For now, grouped view doesn't maintain its own selection state
        # It relies on the ClusterDialog to manage selections
        pass