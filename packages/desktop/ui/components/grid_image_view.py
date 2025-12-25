"""
Grid image view component for displaying multiple images in a flow layout.
"""

from typing import Callable, Dict, List, Optional

from PySide6.QtWidgets import QWidget, QScrollArea
from PySide6.QtCore import Qt, Signal, QTimer

from .layouts import FlowLayout
from .image_widgets import ImageCard, StackImageCard


class GridImageView(QWidget):
    """Widget for displaying images in a responsive grid layout."""

    # Signal emitted when image selection changes (image_path, is_selected)
    selection_changed = Signal(str, bool)
    # Signal emitted during batch creation (current_count, total_count)
    batch_progress = Signal(int, int)

    def __init__(
        self,
        images: List[str],
        parent=None,
        thumbnail_size: int = 150,
        stack_groups: Optional[List[List[str]]] = None,
        stack_open_callback: Optional[Callable[[List[str]], None]] = None,
    ):
        super().__init__(parent)
        self.images = images
        self.cards = []
        self._cards_by_path: Dict[str, QWidget] = {}
        self.thumbnail_size = thumbnail_size
        self.items = self._normalize_items(images, stack_groups)
        self.stack_open_callback = stack_open_callback
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

    def _normalize_items(
        self,
        images: List[str],
        stack_groups: Optional[List[List[str]]],
    ) -> List[List[str]]:
        if stack_groups:
            return [list(group) for group in stack_groups if group]
        return [[path] for path in images]
    
    def create_image_cards_batch(self):
        """Create image cards in batches to prevent UI blocking."""
        if self.current_batch_index >= len(self.items):
            # All cards created - emit final progress
            self.batch_progress.emit(len(self.items), len(self.items))
            return
        
        # Calculate batch range
        start_idx = self.current_batch_index
        end_idx = min(start_idx + self.batch_size, len(self.items))
        
        # Emit progress
        self.batch_progress.emit(end_idx, len(self.items))
        
        # Create cards for this batch
        for i in range(start_idx, end_idx):
            group = self.items[i]
            card = self._create_card_for_group(group)
            if not card:
                continue
            self.cards.append(card)
            self.scroll_layout.addWidget(card)
        
        # Update index for next batch
        self.current_batch_index = end_idx
        
        # Schedule next batch if there are more images
        if self.current_batch_index < len(self.items):
            # Use QTimer to schedule next batch on next event loop iteration
            # This allows UI to remain responsive
            if not self.batch_timer:
                self.batch_timer = QTimer()
                self.batch_timer.setSingleShot(True)
                self.batch_timer.timeout.connect(self.create_image_cards_batch)
            self.batch_timer.start(10)  # Small delay to let UI process events

    def _create_card_for_group(self, group: List[str]) -> Optional[QWidget]:
        if not group:
            return None
        if len(group) == 1:
            image_path = group[0]
            card = ImageCard(image_path, size=self.thumbnail_size)
            card.selection_changed.connect(self.on_card_selection_changed)
            self._cards_by_path[image_path] = card
            return card

        card = StackImageCard(
            group,
            size=self.thumbnail_size,
            open_stack_callback=self.stack_open_callback,
        )
        card.selection_changed.connect(self.on_card_selection_changed)
        for image_path in group:
            self._cards_by_path[image_path] = card
        return card

    def on_card_selection_changed(self, image_path: str, is_selected: bool):
        """Handle selection changes from individual cards."""
        self.selection_changed.emit(image_path, is_selected)

    def set_all_selected(self, selected: bool):
        """Set selection state for all images."""
        for card in self.cards:
            if isinstance(card, StackImageCard):
                card.set_all_selected(selected)
                continue
            if card.is_selected != selected:
                # Temporarily disconnect to avoid signal spam
                card.selection_changed.disconnect()
                card.set_selected(selected)
                card.selection_changed.connect(self.on_card_selection_changed)

    def set_image_selected(self, image_path: str, selected: bool):
        """Set selection state for a specific image."""
        card = self._cards_by_path.get(image_path)
        if not card:
            return
        if isinstance(card, StackImageCard):
            card.set_image_selected(image_path, selected)
            return
        if card.is_selected != selected:
            # Temporarily disconnect to avoid recursion
            card.selection_changed.disconnect()
            card.set_selected(selected)
            card.selection_changed.connect(self.on_card_selection_changed)

    def get_selected_images(self) -> List[str]:
        """Get list of selected image paths."""
        selected = []
        for card in self.cards:
            if isinstance(card, StackImageCard):
                selected.extend(card.get_selected_images())
            elif card.is_selected:
                selected.append(card.image_path)
        return selected
    
    def cleanup(self):
        """Cleanup resources when view is being destroyed."""
        # Stop batch timer if running
        if self.batch_timer and self.batch_timer.isActive():
            self.batch_timer.stop()
        
        # Disconnect all signals to prevent updates during cleanup
        for card in self.cards:
            try:
                card.selection_changed.disconnect()
                if hasattr(card, "cleanup"):
                    card.cleanup()
                elif hasattr(card, "image_widget") and hasattr(card.image_widget, "cleanup"):
                    card.image_widget.cleanup()
            except:
                pass
        # Clear the list
        self.cards.clear()
        self._cards_by_path.clear()
