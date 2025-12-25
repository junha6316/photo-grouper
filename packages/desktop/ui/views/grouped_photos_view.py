"""
Grouped photos view component - wrapper for the existing PreviewPanel.
"""

from typing import List, Dict
import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QLabel,
    QStackedWidget,
    QCheckBox,
    QHBoxLayout,
)
from PySide6.QtCore import Signal, Qt

from ui.preview_panel import PreviewPanel


class GroupedPhotosView(QWidget):
    """View for displaying grouped photos using the existing PreviewPanel."""
    
    selection_changed = Signal(str, bool)  # image_path, is_selected
    group_clicked = Signal(list, int, bool, object)  # images, group_number, is_singles, similarity
    exclude_similar_toggled = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_groups: List[List[str]] = []
        self.main_window = parent
        self.summary_label = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(6, 6, 6, 4)
        self.exclude_similar_checkbox = QCheckBox("Exclude >=0.98 similar")
        self.exclude_similar_checkbox.setStyleSheet("font-size: 11px; color: #555;")
        self.exclude_similar_checkbox.toggled.connect(self.exclude_similar_toggled.emit)
        controls_layout.addWidget(self.exclude_similar_checkbox)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Use QStackedWidget to properly switch between views
        self.stacked_widget = QStackedWidget()
        
        # Create preview panel with scroll area
        self.preview_panel = PreviewPanel()
        self.preview_panel.group_clicked.connect(self.on_group_clicked)
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

        self.summary_label = QLabel("")
        self.summary_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.summary_label.setStyleSheet(
            "color: #666; font-size: 11px; padding: 6px 8px; border-top: 1px solid #eee;"
        )
        self.summary_label.setVisible(False)
        layout.addWidget(self.summary_label)
    
    def set_groups(
        self,
        groups: List[List[str]],
        min_display_size: int = 2,
        similarities: List[float] = None,
        preserve_order: bool = False,
    ):
        """Set the photo groups to display."""
        self.current_groups = groups
        # Switch to preview panel
        self.stacked_widget.setCurrentIndex(0)
        self.preview_panel.display_groups(
            groups,
            min_display_size,
            similarities,
            preserve_order=preserve_order,
        )
    
    def show_processing_message(self, message: str):
        """Show a processing message instead of the preview panel."""
        self.processing_label.setText(f"⏳ {message}")
        # Switch to processing label
        self.stacked_widget.setCurrentIndex(1)
        self.set_summary_text("")
    
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
        self.set_summary_text("")
    
    def get_preview_panel(self) -> PreviewPanel:
        """Get the underlying preview panel."""
        return self.preview_panel

    def set_summary_text(self, text: str):
        """Update the summary text shown below the group list."""
        if not self.summary_label:
            return
        self.summary_label.setText(text)
        self.summary_label.setVisible(bool(text))

    def sync_selected_counts(self, selected_images):
        """Sync selected counts in the group list."""
        if self.preview_panel:
            self.preview_panel.update_selected_counts(selected_images)

    def on_group_clicked(self, images: List[str], group_number: int, is_singles_group: bool, similarity):
        """Handle group click from preview panel."""
        self.group_clicked.emit(images, group_number, is_singles_group, similarity)
    
    def sync_selection(self, image_path: str, is_selected: bool):
        """Sync selection state from other views."""
        # For now, grouped view doesn't maintain its own selection state
        # It relies on the ClusterDialog to manage selections
        pass
