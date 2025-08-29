"""
Grouped photos view component - wrapper for the existing PreviewPanel.
"""

from typing import List, Dict
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PySide6.QtCore import Signal

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
        
        # Create preview panel with scroll area
        self.preview_panel = PreviewPanel()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.preview_panel)
        self.scroll_area.setWidgetResizable(True)
        
        layout.addWidget(self.scroll_area)
    
    def set_groups(self, groups: List[List[str]], min_display_size: int = 2, similarities: List[float] = None):
        """Set the photo groups to display."""
        self.current_groups = groups
        self.preview_panel.display_groups(groups, min_display_size, similarities)
    
    def clear(self):
        """Clear the display."""
        self.current_groups = []
        self.preview_panel.clear()
    
    def get_preview_panel(self) -> PreviewPanel:
        """Get the underlying preview panel."""
        return self.preview_panel
    
    def sync_selection(self, image_path: str, is_selected: bool):
        """Sync selection state from other views."""
        # For now, grouped view doesn't maintain its own selection state
        # It relies on the ClusterDialog to manage selections
        pass