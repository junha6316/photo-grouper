"""
Refactored cluster dialog using modular components.
"""

from typing import List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QStackedWidget, QButtonGroup
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut

from ui.components.grid_image_view import GridImageView
from ui.components.single_image_view import SingleImageView


class ClusterDialog(QDialog):
    """Dialog to show detailed view of a photo cluster with selection support."""
    
    def __init__(self, cluster_images: List[str], cluster_number: int, parent=None):
        super().__init__(parent)
        self.cluster_images = cluster_images
        self.cluster_number = cluster_number
        self.selected_count = 0
        self.current_view_mode = "grid"  # "grid" or "single"
        self.main_window = parent  # Reference to main window for global selection sync
        self._closing = False  # Flag to prevent duplicate cleanup
        self._cleanup_done = False  # Flag to track cleanup completion
        
        self.setWindowTitle(f"Cluster {cluster_number} - {len(cluster_images)} Images")
        self.setGeometry(100, 100, 1000, 700)
        
        self.init_ui()
        self.setup_keyboard_shortcuts()
        self.sync_with_global_selection()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Header with view mode toggles and selection controls
        header_layout = QHBoxLayout()
        
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        
        title_label = QLabel(f"Cluster {self.cluster_number}")
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
        """)
        self.select_all_button.clicked.connect(self.toggle_select_all)
        header_layout.addWidget(self.select_all_button)
        
        info_label = QLabel(f"{len(self.cluster_images)} images")
        info_label.setStyleSheet("color: #666; font-size: 12px;")
        header_layout.addWidget(info_label)
        
        layout.addLayout(header_layout)
        
        # Stacked widget to hold both view modes
        self.stacked_widget = QStackedWidget()
        
        # Create grid view using GridImageView
        self.grid_view = GridImageView(self.cluster_images, self, thumbnail_size=120)
        self.grid_view.selection_changed.connect(self.on_image_selection_changed)
        self.stacked_widget.addWidget(self.grid_view)
        
        # Create single image view
        self.single_view = SingleImageView(self.cluster_images, self)
        self.single_view.selection_changed.connect(self.on_single_view_selection_changed)
        self.stacked_widget.addWidget(self.single_view)
        
        layout.addWidget(self.stacked_widget)
        
        # Action buttons
        button_layout = QHBoxLayout()        
        button_layout.addStretch()
        
        # Export selected button
        self.export_button = QPushButton("📁 Export Selected")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_selected_images)
        self.export_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 12px;
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
        button_layout.addWidget(self.export_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(lambda: self.accept())  # Direct close without blocking
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
        elif mode == "single":
            self.stacked_widget.setCurrentIndex(1)
            self.single_view_button.setChecked(True)
            # Sync selections from grid to single view
            self.sync_selections_to_single_view()
    
    def sync_selections_to_single_view(self):
        """Sync selections from grid view to single view."""
        selected_images = self.grid_view.get_selected_images()
        for image_path in selected_images:
            self.single_view.set_selection(image_path, True)
    
    def sync_selections_to_grid_view(self):
        """Sync selections from single view to grid view."""
        selected_images = self.single_view.get_selected_images()
        for image_path in self.cluster_images:
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
    
    def toggle_view_mode(self):
        """Toggle between grid and single view modes."""
        if self.current_view_mode == "grid":
            self.set_view_mode("single")
        else:
            self.set_view_mode("grid")
    
    def on_single_view_selection_changed(self, image_path: str, is_selected: bool):
        """Handle selection changes from single view."""
        # Update the corresponding image in grid view
        self.grid_view.set_image_selected(image_path, is_selected)
    
    def on_image_selection_changed(self, image_path: str, is_selected: bool):
        """Handle individual image selection changes."""
        # Update selection count and UI
        if is_selected:
            self.selected_count += 1
        else:
            self.selected_count -= 1
        
        # Also update single view selection if it exists
        if hasattr(self, 'single_view'):
            self.single_view.set_selection(image_path, is_selected)
        
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
        
        # Update export button state
        self.export_button.setEnabled(self.selected_count > 0)
        
        if self.selected_count == 0:
            self.select_all_button.setText("Select All")
        
        elif self.selected_count == len(self.cluster_images):
            self.select_all_button.setText("Deselect All")
        
        else:
            self.select_all_button.setText("Select All")
        
    
    def toggle_select_all(self):
        """Toggle selection of all images in the cluster."""
        all_selected = self.selected_count == len(self.cluster_images)
        target_state = not all_selected
        
        self.grid_view.set_all_selected(target_state)
        
        print(f"{'Selected' if target_state else 'Deselected'} all images in "
              f"cluster {self.cluster_number}")
    
    def show_selected_photos(self):
        """Show selected photos in a separate dialog."""
        selected_images = self.get_selected_images()
        if selected_images:
            from ui.selected_view_dialog import SelectedViewDialog
            dialog = SelectedViewDialog(selected_images, self)
            dialog.exec()
    
    def sync_with_global_selection(self):
        """Sync this cluster's selection state with global selection."""
        if not self.main_window or not hasattr(self.main_window, 'get_global_selected_images'):
            return
        
        global_selected = self.main_window.get_global_selected_images()
        
        # Reset count and recalculate
        self.selected_count = 0
        
        # Update grid view to match global selection
        for image_path in self.cluster_images:
            should_be_selected = image_path in global_selected
            self.grid_view.set_image_selected(image_path, should_be_selected)
            
            # Count selected items
            if should_be_selected:
                self.selected_count += 1
        
        # Update single view if it exists
        if hasattr(self, 'single_view'):
            for image_path in self.cluster_images:
                should_be_selected = image_path in global_selected
                self.single_view.set_selection(image_path, should_be_selected)
        
        # Update UI
        self.update_selection_ui()
    
    def get_selected_images(self) -> List[str]:
        """Get list of selected image paths."""
        # Use grid view selection as the source of truth
        return self.grid_view.get_selected_images()
    
    def export_selected_images(self):
        """Open export dialog for selected images in this cluster."""
        selected_images = self.get_selected_images()
        if not selected_images:
            return
        
        from ui.export_dialog import ExportDialog
        dialog = ExportDialog(selected_images, self)
        dialog.exec()
    
    def close_dialog(self):
        """Close dialog immediately without blocking."""
        # Store state for deferred cleanup
        self._closing = True
        
        # Close immediately
        self.accept()
    
    def closeEvent(self, event):
        """Handle close event with optimized cleanup."""
        # Mark as closing
        self._closing = True
        
        # Accept the close event immediately to avoid lag
        event.accept()
        
        # Perform minimal cleanup after dialog is hidden
        QTimer.singleShot(10, self._deferred_cleanup)
    
    def _deferred_cleanup(self):
        """Perform cleanup operations after dialog is closed."""
        # Prevent duplicate cleanup
        if self._cleanup_done:
            return
        self._cleanup_done = True
        
        try:
            # Only sync selections if main window exists
            if self.main_window and hasattr(self.main_window, 'add_to_global_selection'):
                # Sync selections from single view if needed
                if hasattr(self, 'current_view_mode') and self.current_view_mode == "single":
                    self.sync_selections_to_grid_view()
                
                # Get final selection state
                selected_images = set(self.get_selected_images())
                
                # Batch update global selection
                for image_path in self.cluster_images:
                    if image_path in selected_images:
                        self.main_window.add_to_global_selection(image_path)
                    else:
                        self.main_window.remove_from_global_selection(image_path)
            
            # Cleanup grid view resources
            if hasattr(self, 'grid_view') and hasattr(self.grid_view, 'cleanup'):
                self.grid_view.cleanup()
            
            # Cleanup single view if exists
            if hasattr(self, 'single_view') and hasattr(self.single_view, 'cleanup'):
                self.single_view.cleanup()
        except:
            pass  # Ignore errors during cleanup