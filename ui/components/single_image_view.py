"""
Single image view component with navigation and selection.
"""

import os
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton
)
from PySide6.QtCore import Qt, Signal

from .thumbnail_strip import ThumbnailStrip
from .panels import SelectedImagesPanel
from ..utils.image_utils import load_image_as_pixmap


class SingleImageView(QWidget):
    """Widget for viewing a single image with navigation controls and selected images panel."""
    
    # Signals
    image_changed = Signal(int)  # current_index
    selection_changed = Signal(str, bool)  # image_path, is_selected
    
    def __init__(self, images: List[str], parent=None):
        super().__init__(parent)
        self.images = images
        self.current_index = 0
        self.selected_images = set()
        
        self.init_ui()
        self.update_image()
    
    def init_ui(self):
        """Initialize the single image view UI."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Left side - main image view
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Navigation header
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:hover:enabled {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
            }
        """)
        nav_layout.addWidget(self.prev_button)
        
        nav_layout.addStretch()
        
        self.image_counter = QLabel()
        self.image_counter.setAlignment(Qt.AlignCenter)
        self.image_counter.setStyleSheet(
            "font-size: 14px; color: #666; font-weight: bold;"
        )
        nav_layout.addWidget(self.image_counter)
        
        nav_layout.addStretch()
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:hover:enabled {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
            }
        """)
        nav_layout.addWidget(self.next_button)
        
        left_layout.addLayout(nav_layout)
        
        # Main image area
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: #fafafa;
            }
        """)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: none;
                padding: 10px;
            }
        """)
        
        self.image_scroll.setWidget(self.image_label)
        left_layout.addWidget(self.image_scroll, 1)
        
        # Image info and controls
        info_layout = QHBoxLayout()
        
        self.filename_label = QLabel()
        self.filename_label.setStyleSheet("font-size: 12px; color: #666;")
        info_layout.addWidget(self.filename_label)
        
        info_layout.addStretch()
        
        self.select_button = QPushButton("Select")
        self.select_button.setCheckable(True)
        self.select_button.clicked.connect(self.toggle_current_selection)
        self.select_button.setStyleSheet("""
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
            QPushButton:checked {
                background-color: #007acc;
                color: white;
                border-color: #005599;
            }
        """)
        info_layout.addWidget(self.select_button)
        
        left_layout.addLayout(info_layout)
        
        # Thumbnail strip at bottom
        self.thumbnail_strip = ThumbnailStrip(self.images, height=110, thumbnail_size=120)
        self.thumbnail_strip.thumbnail_clicked.connect(self.on_thumbnail_clicked)
        left_layout.addWidget(self.thumbnail_strip, 0)
        
        # Add left layout to main horizontal layout with weight 3
        main_layout.addLayout(left_layout, 3)
        
        # Right side - selected images panel
        self.selected_panel = SelectedImagesPanel(self)
        self.selected_panel.setFixedWidth(200)  # Fixed width for selected images panel
        main_layout.addWidget(self.selected_panel, 0)
        
        # Connect selected panel removal to selection system
        def on_remove_from_selected(image_path):
            if image_path in self.selected_images:
                self.selected_images.remove(image_path)
                # Update current image UI if it matches
                if (self.images and self.current_index < len(self.images) and 
                    self.images[self.current_index] == image_path):
                    self.select_button.setChecked(False)
                    self.select_button.setText("Select")
                # Emit signal to update global selection
                self.selection_changed.emit(image_path, False)
        
        # Store the connection function for later use when thumbnails are created
        self.selected_panel._connect_removal_handler = on_remove_from_selected
    
    
    def on_thumbnail_clicked(self, index: int):
        """Handle thumbnail click to change current image."""
        if 0 <= index < len(self.images):
            self.current_index = index
            self.update_image()
    
    def update_image(self):
        """Update the displayed image and UI elements."""
        if not self.images or self.current_index >= len(self.images):
            return
        
        current_image = self.images[self.current_index]
        
        # Update navigation buttons
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.images) - 1)
        
        # Update counter
        self.image_counter.setText(f"{self.current_index + 1} / {len(self.images)}")
        
        # Update filename
        filename = os.path.basename(current_image)
        self.filename_label.setText(f"📷 {filename}")
        
        # Update selection state
        is_selected = current_image in self.selected_images
        self.select_button.setChecked(is_selected)
        self.select_button.setText("Deselect" if is_selected else "Select")
        
        # Load and display image
        self.load_large_image(current_image)
        
        # Update thumbnail strip to show current selection
        self.thumbnail_strip.set_current_index(self.current_index)
        
        # Emit signal
        self.image_changed.emit(self.current_index)
    
    def load_large_image(self, image_path: str):
        """Load and display a large version of the image."""
        try:
            # Show loading indicator
            self.image_label.setText("📷 Loading...")
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: white;
                    border: none;
                    padding: 10px;
                    color: #666;
                    font-size: 14px;
                }
            """)
            
            # Load image directly without using shared async loader to avoid conflicts
            self._load_image_directly(image_path)
                
        except Exception as e:
            # Show error
            self.image_label.setText(f"⚠️ Error loading image:\n{str(e)}")
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: white;
                    border: none;
                    padding: 10px;
                    color: red;
                    font-size: 12px;
                }
            """)
    
    def _load_image_directly(self, image_path: str):
        """Load image directly without using the shared async loader."""
        # Use the centralized image loading utility
        max_size = 700  # Maximum dimension for single view
        pixmap = load_image_as_pixmap(image_path, max_size=max_size)
        
        if pixmap:
            # Display the image
            self.image_label.setPixmap(pixmap)
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: white;
                    border: none;
                    padding: 10px;
                }
            """)
        else:
            # Show error
            self.image_label.setText(f"⚠️ Error loading image")
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: white;
                    border: none;
                    padding: 10px;
                    color: red;
                    font-size: 12px;
                }
            """)
    
    def prev_image(self):
        """Go to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()
    
    def next_image(self):
        """Go to next image."""
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.update_image()
    
    def go_to_image(self, index: int):
        """Go to specific image index."""
        if 0 <= index < len(self.images):
            self.current_index = index
            self.update_image()
    
    def toggle_current_selection(self):
        """Toggle selection of current image."""
        if not self.images or self.current_index >= len(self.images):
            return
        
        current_image = self.images[self.current_index]
        is_selected = current_image in self.selected_images
        
        if is_selected:
            self.selected_images.remove(current_image)
            self.selected_panel.remove_selected_image(current_image)
        else:
            self.selected_images.add(current_image)
            self.selected_panel.add_selected_image(current_image)
        
        # Update UI
        self.select_button.setChecked(not is_selected)
        self.select_button.setText("Deselect" if not is_selected else "Select")
        
        # Emit signal (this will trigger on_single_view_selection_changed which handles global update)
        self.selection_changed.emit(current_image, not is_selected)
    
    def set_selection(self, image_path: str, selected: bool):
        """Set selection state for an image."""
        if selected:
            self.selected_images.add(image_path)
            self.selected_panel.add_selected_image(image_path)
        else:
            self.selected_images.discard(image_path)
            self.selected_panel.remove_selected_image(image_path)
        
        # Update UI if this is the current image
        if (self.images and self.current_index < len(self.images) and 
            self.images[self.current_index] == image_path):
            self.select_button.setChecked(selected)
            self.select_button.setText("Deselect" if selected else "Select")
    
    def get_selected_images(self):
        """Get set of selected image paths."""
        return self.selected_images.copy()