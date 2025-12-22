"""
Grid image view component for displaying multiple images in a flow layout.
"""

from typing import List

from PySide6.QtWidgets import QWidget, QScrollArea
from PySide6.QtCore import Qt, Signal, QTimer

from .layouts import FlowLayout
from .image_widgets import ImageCard


class GridImageView(QWidget):
    """Widget for displaying images in a responsive grid layout."""

    # Signal emitted when image selection changes (image_path, is_selected)
    selection_changed = Signal(str, bool)
    # Signal emitted during batch creation (current_count, total_count)
    batch_progress = Signal(int, int)

    def __init__(self, images: List[str], parent=None, thumbnail_size: int = 150):
        super().__init__(parent)
        self.images = images
        self.image_cards = []
        self.thumbnail_size = thumbnail_size
        self.current_batch_index = 0
        self.batch_size = 20  # Create cards in batches of 20 to prevent UI blocking
        self.batch_timer = None

        self.init_ui()
        self.create_image_cards_batch()  # Start batch creation instead of all at once

    def init_ui(self):
        """Initialize the grid view UI."""
        # Images scroll area with responsive flow layout
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #fafafa;
            }
        """)

        # Main container for flow layout
        self.scroll_widget = QWidget()
        self.scroll_layout = FlowLayout(self.scroll_widget)
        self.scroll_layout.setSpacing(15)  # Increased from 10 to 15
        self.scroll_layout.setContentsMargins(20, 20, 20, 20)  # Increased from 15 to 20

        self.scroll_area.setWidget(self.scroll_widget)

        # Set up main layout
        from PySide6.QtWidgets import QVBoxLayout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll_area)

    def create_image_cards(self):
        """Create image cards for each image (legacy method for compatibility)."""
        for image_path in self.images:
            card = ImageCard(image_path, size=self.thumbnail_size)
            card.selection_changed.connect(self.on_card_selection_changed)
            self.image_cards.append(card)
            self.scroll_layout.addWidget(card)
    
    def create_image_cards_batch(self):
        """Create image cards in batches to prevent UI blocking."""
        if self.current_batch_index >= len(self.images):
            # All cards created - emit final progress
            self.batch_progress.emit(len(self.images), len(self.images))
            return
        
        # Calculate batch range
        start_idx = self.current_batch_index
        end_idx = min(start_idx + self.batch_size, len(self.images))
        
        # Emit progress
        self.batch_progress.emit(end_idx, len(self.images))
        
        # Create cards for this batch
        for i in range(start_idx, end_idx):
            image_path = self.images[i]
            card = ImageCard(image_path, size=self.thumbnail_size)
            card.selection_changed.connect(self.on_card_selection_changed)
            self.image_cards.append(card)
            self.scroll_layout.addWidget(card)
        
        # Update index for next batch
        self.current_batch_index = end_idx
        
        # Schedule next batch if there are more images
        if self.current_batch_index < len(self.images):
            # Use QTimer to schedule next batch on next event loop iteration
            # This allows UI to remain responsive
            if not self.batch_timer:
                self.batch_timer = QTimer()
                self.batch_timer.setSingleShot(True)
                self.batch_timer.timeout.connect(self.create_image_cards_batch)
            self.batch_timer.start(10)  # Small delay to let UI process events

    def on_card_selection_changed(self, image_path: str, is_selected: bool):
        """Handle selection changes from individual cards."""
        self.selection_changed.emit(image_path, is_selected)

    def set_all_selected(self, selected: bool):
        """Set selection state for all images."""
        for card in self.image_cards:
            if card.is_selected != selected:
                # Temporarily disconnect to avoid signal spam
                card.selection_changed.disconnect()
                card.set_selected(selected)
                card.selection_changed.connect(self.on_card_selection_changed)

    def set_image_selected(self, image_path: str, selected: bool):
        """Set selection state for a specific image."""
        for card in self.image_cards:
            if card.image_path == image_path:
                if card.is_selected != selected:
                    # Temporarily disconnect to avoid recursion
                    card.selection_changed.disconnect()
                    card.set_selected(selected)
                    card.selection_changed.connect(
                        self.on_card_selection_changed)
                break

    def get_selected_images(self) -> List[str]:
        """Get list of selected image paths."""
        return [card.image_path for card in self.image_cards
                if card.is_selected]

    def get_image_cards(self) -> List[ImageCard]:
        """Get list of all image cards."""
        return self.image_cards.copy()
    
    def cleanup(self):
        """Cleanup resources when view is being destroyed."""
        # Stop batch timer if running
        if self.batch_timer and self.batch_timer.isActive():
            self.batch_timer.stop()
        
        # Disconnect all signals to prevent updates during cleanup
        for card in self.image_cards:
            try:
                card.selection_changed.disconnect()
                # Call cleanup on image widget if it has one
                if hasattr(card.image_widget, 'cleanup'):
                    card.image_widget.cleanup()
            except:
                pass
        # Clear the list
        self.image_cards.clear()
