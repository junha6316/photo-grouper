"""
Image widget components for displaying and interacting with images.
"""

import os
from typing import Callable, List

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor

from ui.utils.async_image_loader import get_async_loader, ImageLoadResult


class ShadowCheckBox(QCheckBox):
    """Checkbox with a drop shadow to emulate CSS box-shadow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hovered = False
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setOffset(0, 2)
        self.setGraphicsEffect(self._shadow)
        self.toggled.connect(lambda _checked: self._update_shadow())
        self._update_shadow()

    def enterEvent(self, event):
        self._hovered = True
        self._update_shadow()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._update_shadow()
        super().leaveEvent(event)

    def _update_shadow(self):
        if self.isChecked():
            if self._hovered:
                color = QColor(0, 85, 153, 128)
                blur = 8
            else:
                color = QColor(0, 122, 204, 102)
                blur = 6
        else:
            if self._hovered:
                color = QColor(0, 122, 204, 77)
                blur = 6
            else:
                color = QColor(0, 0, 0, 51)
                blur = 4

        self._shadow.setBlurRadius(blur)
        self._shadow.setColor(color)
        self._shadow.setOffset(0, 2)


class ImageCard(QWidget):
    """A card widget containing an image and filename label with selection support."""
    
    # Signal emitted when selection state changes
    selection_changed = Signal(str, bool)  # image_path, is_selected
    
    def __init__(self, image_path: str, size: int = 150):
        super().__init__()
        self.image_path = image_path
        self.thumbnail_size = size
        self.is_selected = False
        
        self.setFixedSize(size + 20, size + 40)  # Extra space for filename
        # Set object name for CSS targeting
        self.setObjectName("imageCard")
        # Ensure stylesheet background/border rendering on a plain QWidget
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Container for image with checkbox overlay
        image_container = QWidget()
        image_container.setFixedSize(self.thumbnail_size + 10, self.thumbnail_size + 10)
        
        # Image widget
        self.image_widget = ImageWidget(self.image_path, self.thumbnail_size)
        self.image_widget.setParent(image_container)
        self.image_widget.move(0, 0)
        
        # Checkbox overlay in top-right corner - increased size for better visibility
        self.checkbox = ShadowCheckBox()
        self.checkbox.setParent(image_container)
        self.checkbox.setFixedSize(36, 36)  # Increased from 28 to 36
        self.checkbox.move(self.thumbnail_size - 26, 5)  # Adjusted position
        self.checkbox.setChecked(self.is_selected)
        self.checkbox.toggled.connect(self.on_checkbox_toggled)
        self.checkbox.setCursor(Qt.PointingHandCursor)

        # Enhanced checkbox style with larger, more visible design
        checkbox_style = """
            QCheckBox {
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 30px;
                height: 30px;
                border-radius: 6px;
                border: 2.5px solid rgba(0, 0, 0, 0.4);
                background-color: rgba(255, 255, 255, 0.98);
            }
            QCheckBox::indicator:unchecked:hover {
                border: 2.5px solid #007acc;
                background-color: rgba(240, 248, 255, 0.98);
            }
            QCheckBox::indicator:checked {
                border: 2.5px solid #007acc;
                background-color: #007acc;
                color: white;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #005599;
                border: 2.5px solid #005599;
            }
        """

        self.checkbox.setStyleSheet(checkbox_style)

        # Add checkmark overlay label (only visible when checked) - larger size
        self.checkmark = QLabel("✓")
        self.checkmark.setParent(self.checkbox)
        self.checkmark.setFixedSize(30, 30)  # Increased from 22 to 30
        self.checkmark.move(3, 3)  # Adjusted position
        self.checkmark.setAlignment(Qt.AlignCenter)
        self.checkmark.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 22px;
                font-weight: bold;
                background-color: transparent;
            }
        """)
        self.checkmark.setVisible(self.is_selected)
        self.checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        layout.addWidget(image_container)
        
        # Filename label
        filename = os.path.basename(self.image_path)
        if len(filename) > 20:
            filename = filename[:17] + "..."
        
        self.filename_label = QLabel(filename)
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setToolTip(os.path.basename(self.image_path))
        layout.addWidget(self.filename_label)
        
        # Apply initial styling
        self._update_style()
    
    def _update_style(self):
        """Update widget style based on selection state."""
        if self.is_selected:
            # Use object name selector for more reliable targeting
            self.setStyleSheet("""
                #imageCard {
                    background-color: #e6f3ff;
                    border-radius: 16px;
                    border: 2px solid #007acc;
                }
                #imageCard:hover {
                    background-color: #cce7ff;
                    border-color: #005599;
                }
                #imageCard QLabel {
                    color: #555;
                    font-size: 11px;
                    background: transparent;
                    border: none;
                    padding: 2px;
                }
            """)
        else:
            self.setStyleSheet("""
                #imageCard {
                    background-color: #fafafa;
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                }
                #imageCard:hover {
                    background-color: #f0f8ff;
                    border-color: #007acc;
                }
                #imageCard QLabel {
                    color: #555;
                    font-size: 11px;
                    background: transparent;
                    border: none;
                    padding: 2px;
                }
            """)
        
        # Force style refresh
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
    
    def on_checkbox_toggled(self, checked: bool):
        """Handle checkbox toggle events."""
        self.checkmark.setVisible(checked)
        self.set_selected(checked)
    
    def set_selected(self, selected: bool):
        """Set selection state."""
        if self.is_selected != selected:
            self.is_selected = selected
            self._update_style()
            # Update checkbox without triggering signal
            self.checkbox.blockSignals(True)
            self.checkbox.setChecked(selected)
            self.checkbox.blockSignals(False)
            # Update checkmark visibility
            self.checkmark.setVisible(selected)
            self.selection_changed.emit(self.image_path, self.is_selected)


class StackThumbnail(QLabel):
    """Thumbnail widget used for stacked duplicate previews."""

    def __init__(self, image_path: str, size: int = 150):
        super().__init__()
        self.image_path = image_path
        self.thumbnail_size = size
        self.loader_connection = None
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 6px;
                background-color: #f9f9f9;
                padding: 2px;
            }
        """)
        self.setText("📷")
        self.load_thumbnail_async()

    def cleanup(self):
        """Cleanup signal connections."""
        if self.loader_connection:
            try:
                loader = get_async_loader()
                loader.image_loaded.disconnect(self.loader_connection)
            except Exception:
                pass
            self.loader_connection = None

    def load_thumbnail_async(self):
        loader = get_async_loader()
        cached_result = loader.get_cached_image(self.image_path, self.thumbnail_size)
        if cached_result:
            self.on_image_loaded(cached_result)
            return

        self.loader_connection = loader.image_loaded.connect(self.on_image_loaded)
        loader.load_image(self.image_path, self.thumbnail_size, priority=1)

    def on_image_loaded(self, result: ImageLoadResult):
        if result.image_path != self.image_path:
            return
        if self.loader_connection:
            loader = get_async_loader()
            loader.image_loaded.disconnect(self.loader_connection)
            self.loader_connection = None

        if result.success and result.pixmap:
            pixmap = result.pixmap.scaled(
                self.thumbnail_size,
                self.thumbnail_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            pixmap.setDevicePixelRatio(1.0)
            self.setPixmap(pixmap)
            self.setToolTip(f"{os.path.basename(self.image_path)}\n{self.image_path}")
        else:
            filename = os.path.basename(self.image_path)
            if len(filename) > 15:
                filename = filename[:12] + "..."
            self.setText(f"⚠️\n{filename}")
            self.setStyleSheet(self.styleSheet() + """
                QLabel {
                    color: red;
                    font-size: 9px;
                }
            """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            parent_widget = self.parent()
            while parent_widget:
                if isinstance(parent_widget, StackImageCard):
                    parent_widget.open_stack_dialog()
                    break
                parent_widget = parent_widget.parent()
            event.accept()
            return
        super().mousePressEvent(event)


class StackImageCard(QWidget):
    """Card widget representing a stack of near-duplicate images."""

    selection_changed = Signal(str, bool)

    def __init__(
        self,
        image_paths: List[str],
        size: int = 150,
        max_stack: int = 3,
        offset: int = 5,
        open_stack_callback: Callable[[List[str]], None] = None,
    ):
        super().__init__()
        self.image_paths = list(dict.fromkeys(image_paths))
        self.thumbnail_size = size
        self.max_stack = max_stack
        self.offset = offset
        self.selected_images = set()
        self.open_stack_callback = open_stack_callback

        self.setFixedSize(size + 20, size + 40)
        self.setObjectName("stackImageCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)

        self.checkbox = None
        self.checkmark = None
        self.stack_thumbnails = []
        self.filename_label = None

        self.init_ui()
        self._update_style()
        self._update_checkbox_state()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        image_container = QWidget()
        image_container.setFixedSize(self.thumbnail_size + 10, self.thumbnail_size + 10)

        stack_paths = self.image_paths[: self.max_stack]
        for idx, image_path in enumerate(reversed(stack_paths)):
            thumb = StackThumbnail(image_path, size=self.thumbnail_size)
            thumb.setParent(image_container)
            thumb.move(self.offset * idx, self.offset * idx)
            thumb.show()
            self.stack_thumbnails.append(thumb)

        self.checkbox = ShadowCheckBox()
        self.checkbox.setParent(image_container)
        self.checkbox.setFixedSize(36, 36)
        self.checkbox.move(self.thumbnail_size - 26, 5)
        self.checkbox.setTristate(True)
        self.checkbox.clicked.connect(self.toggle_all_selection)
        self.checkbox.setCursor(Qt.PointingHandCursor)

        self.checkbox.setStyleSheet("""
            QCheckBox {
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 30px;
                height: 30px;
                border-radius: 6px;
                border: 2.5px solid rgba(0, 0, 0, 0.4);
                background-color: rgba(255, 255, 255, 0.98);
            }
            QCheckBox::indicator:unchecked:hover {
                border: 2.5px solid #007acc;
                background-color: rgba(240, 248, 255, 0.98);
            }
            QCheckBox::indicator:checked {
                border: 2.5px solid #007acc;
                background-color: #007acc;
                color: white;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #005599;
                border: 2.5px solid #005599;
            }
            QCheckBox::indicator:indeterminate {
                border: 2.5px solid #999;
                background-color: #ddd;
            }
        """)

        self.checkmark = QLabel("✓")
        self.checkmark.setParent(self.checkbox)
        self.checkmark.setFixedSize(30, 30)
        self.checkmark.move(3, 3)
        self.checkmark.setAlignment(Qt.AlignCenter)
        self.checkmark.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
                background-color: transparent;
            }
        """)
        self.checkmark.setVisible(False)
        self.checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)

        badge = QLabel(f"x{len(self.image_paths)}", image_container)
        badge.setAttribute(Qt.WA_StyledBackground, True)
        badge.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 0.65);
                color: white;
                font-size: 10px;
                padding: 2px 6px;
                border-radius: 9px;
            }
        """)
        badge.adjustSize()
        badge.move(4, 4)

        layout.addWidget(image_container)

        label_text = self._stack_label_text()
        self.filename_label = QLabel(label_text)
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setToolTip(label_text)
        layout.addWidget(self.filename_label)

    def cleanup(self):
        for thumb in self.stack_thumbnails:
            thumb.cleanup()
        self.stack_thumbnails.clear()
        if self.checkbox:
            try:
                self.checkbox.clicked.disconnect()
            except Exception:
                pass

    def _stack_label_text(self) -> str:
        if not self.image_paths:
            return ""
        base = os.path.basename(self.image_paths[0])
        extra = ""
        if len(self.image_paths) > 1:
            extra = f" +{len(self.image_paths) - 1}"
        max_len = max(0, 20 - len(extra))
        if len(base) > max_len and max_len > 3:
            base = base[: max_len - 3] + "..."
        elif len(base) > max_len:
            base = base[:max_len]
        return f"{base}{extra}"

    def _update_checkbox_state(self):
        if not self.checkbox:
            return
        total = len(self.image_paths)
        selected_count = len(self.selected_images)
        if selected_count == 0:
            state = Qt.Unchecked
            if self.checkmark:
                self.checkmark.setVisible(False)
        elif selected_count == total:
            state = Qt.Checked
            if self.checkmark:
                self.checkmark.setText("✓")
                self.checkmark.setVisible(True)
        else:
            state = Qt.PartiallyChecked
            if self.checkmark:
                self.checkmark.setText("-")
                self.checkmark.setVisible(True)
        self.checkbox.blockSignals(True)
        self.checkbox.setCheckState(state)
        self.checkbox.blockSignals(False)

    def _update_style(self):
        total = len(self.image_paths)
        selected_count = len(self.selected_images)
        if selected_count == total and total > 0:
            style = """
                #stackImageCard {
                    background-color: #e6f3ff;
                    border-radius: 16px;
                    border: 2px solid #007acc;
                }
                #stackImageCard:hover {
                    background-color: #cce7ff;
                    border-color: #005599;
                }
                #stackImageCard QLabel {
                    color: #555;
                    font-size: 11px;
                    background: transparent;
                    border: none;
                    padding: 2px;
                }
            """
        elif selected_count > 0:
            style = """
                #stackImageCard {
                    background-color: #f0f7ff;
                    border-radius: 12px;
                    border: 2px dashed #7aa7d9;
                }
                #stackImageCard:hover {
                    background-color: #e0f0ff;
                    border-color: #5c8fc7;
                }
                #stackImageCard QLabel {
                    color: #555;
                    font-size: 11px;
                    background: transparent;
                    border: none;
                    padding: 2px;
                }
            """
        else:
            style = """
                #stackImageCard {
                    background-color: #fafafa;
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                }
                #stackImageCard:hover {
                    background-color: #f0f8ff;
                    border-color: #007acc;
                }
                #stackImageCard QLabel {
                    color: #555;
                    font-size: 11px;
                    background: transparent;
                    border: none;
                    padding: 2px;
                }
            """
        self.setStyleSheet(style)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_image_selected(self, image_path: str, selected: bool):
        if image_path not in self.image_paths:
            return
        if selected:
            self.selected_images.add(image_path)
        else:
            self.selected_images.discard(image_path)
        self._update_checkbox_state()
        self._update_style()

    def set_all_selected(self, selected: bool):
        if selected:
            self.selected_images = set(self.image_paths)
        else:
            self.selected_images.clear()
        self._update_checkbox_state()
        self._update_style()

    def get_selected_images(self) -> List[str]:
        return list(self.selected_images)

    def toggle_all_selection(self):
        total = len(self.image_paths)
        target_state = len(self.selected_images) != total
        previous = set(self.selected_images)
        if target_state:
            self.selected_images = set(self.image_paths)
        else:
            self.selected_images.clear()
        self._update_checkbox_state()
        self._update_style()

        for image_path in self.image_paths:
            is_selected = image_path in self.selected_images
            if (image_path in previous) != is_selected:
                self.selection_changed.emit(image_path, is_selected)

    def open_stack_dialog(self):
        if self.open_stack_callback:
            self.open_stack_callback(self.image_paths)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.open_stack_dialog()
            event.accept()
            return
        super().mousePressEvent(event)


class ImageWidget(QLabel):
    """Widget to display a single image with filename using async loading."""
    
    def __init__(self, image_path: str, size: int = 200):
        super().__init__()
        self.image_path = image_path
        self.thumbnail_size = size
        self.is_loading = True
        self.loader_connection = None  # Track the signal connection
        self.load_timer = None  # Timer for deferred loading
        self.has_loaded = False  # Track if image has been loaded
        
        self.setFixedSize(size + 10, size + 10)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
                padding: 5px;
            }
            QLabel:hover {
                border-color: #007acc;
                background-color: #f0f8ff;
            }
        """)
        
        # Show loading indicator
        self.show_loading()
        
        # Defer loading with a small delay to prevent UI blocking
        # This allows the UI to render first before starting loads
        self.defer_load_image()
    
    def cleanup(self):
        """Cleanup signal connections."""
        # Cancel pending load timer
        if hasattr(self, 'load_timer') and self.load_timer:
            self.load_timer.stop()
            self.load_timer = None
        
        if hasattr(self, 'loader_connection') and self.loader_connection:
            try:
                loader = get_async_loader()
                loader.image_loaded.disconnect(self.loader_connection)
                self.loader_connection = None
            except:
                pass
    
    def show_loading(self):
        """Show loading indicator."""
        self.setText("📷\nLoading...")
        self.setStyleSheet(self.styleSheet() + """
            QLabel { 
                color: #666; 
                font-size: 12px;
            }
        """)
    
    def defer_load_image(self):
        """Defer image loading to prevent UI blocking during initial creation."""
        if not self.has_loaded:
            # Use a timer to defer loading
            if not self.load_timer:
                self.load_timer = QTimer()
                self.load_timer.setSingleShot(True)
                self.load_timer.timeout.connect(self.load_image_async)
            # Random small delay between 50-200ms to stagger loads
            import random
            delay = random.randint(50, 200)
            self.load_timer.start(delay)
    
    def load_image_async(self):
        """Start async image loading."""
        if self.has_loaded:
            return
        
        self.has_loaded = True
        
        # Get global async loader
        loader = get_async_loader()
        
        # First check if image is already in cache
        cached_result = loader.get_cached_image(self.image_path, self.thumbnail_size)
        if cached_result:
            # Image is already cached, use it immediately without async loading
            self.on_image_loaded(cached_result)
            return
        
        # Not in cache, proceed with async loading
        # Connect to the loader's signal and store the connection
        self.loader_connection = loader.image_loaded.connect(self.on_image_loaded)
        
        # Queue the image for loading
        loader.load_image(self.image_path, self.thumbnail_size, priority=1)
    
    def on_image_loaded(self, result: ImageLoadResult):
        """Handle async image loading completion."""
        # Only process results for this specific image
        if result.image_path != self.image_path:
            return
        
        # Disconnect from the loader signal now that we have our image
        if self.loader_connection:
            loader = get_async_loader()
            loader.image_loaded.disconnect(self.loader_connection)
            self.loader_connection = None
        
        self.is_loading = False
        
        if result.success and result.pixmap:
            # Normalize pixmap size to avoid tiny renders from small originals or HiDPI DPR.
            pixmap = result.pixmap.scaled(
                self.thumbnail_size,
                self.thumbnail_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            pixmap.setDevicePixelRatio(1.0)
            self.setPixmap(pixmap)
            self.setToolTip(f"{os.path.basename(self.image_path)}\n{self.image_path}")
            # Reset stylesheet to remove loading text styling
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #ddd;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                    padding: 5px;
                }
                QLabel:hover {
                    border-color: #007acc;
                    background-color: #f0f8ff;
                }
            """)
        else:
            # Failed to load image
            filename = os.path.basename(self.image_path)
            self.setText(f"⚠️\n{filename}")
            self.setStyleSheet(self.styleSheet() + """
                QLabel { 
                    color: red; 
                    font-size: 10px;
                    text-align: center;
                }
            """)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks by forwarding to parent ImageCard."""
        if event.button() == Qt.LeftButton:
            # Find the parent ImageCard and toggle its selection via checkbox
            parent_widget = self.parent()
            while parent_widget:
                if isinstance(parent_widget, ImageCard):
                    # Toggle the checkbox which will trigger selection
                    parent_widget.checkbox.setChecked(not parent_widget.checkbox.isChecked())
                    break
                parent_widget = parent_widget.parent()
        super().mousePressEvent(event)


class ThumbnailWidget(QLabel):
    """Small thumbnail widget for the bottom thumbnail list."""
    
    clicked = Signal(int)  # Emits the index when clicked
    
    def __init__(self, image_path: str, index: int, size: int = 120):
        super().__init__()
        self.image_path = image_path
        self.index = index
        self.thumbnail_size = size
        self.is_current = False
        self.loader_connection = None  # Track the signal connection
        self.has_loaded = False
        self.load_timer = None
        
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 6px;
                background-color: #fafafa;
                padding: 2px;
            }
            QLabel:hover {
                border-color: #007acc;
                background-color: #f0f8ff;
            }
        """)
        
        # Show loading indicator
        self.setText("📷")
        
        # Defer loading with a small delay
        self.defer_load_thumbnail()
    
    def cleanup(self):
        """Cleanup signal connections."""
        if hasattr(self, 'load_timer') and self.load_timer:
            self.load_timer.stop()
            self.load_timer = None
        
        if hasattr(self, 'loader_connection') and self.loader_connection:
            try:
                loader = get_async_loader()
                loader.image_loaded.disconnect(self.loader_connection)
                self.loader_connection = None
            except:
                pass
    
    def defer_load_thumbnail(self):
        """Defer thumbnail loading to prevent UI blocking."""
        if not self.has_loaded:
            if not self.load_timer:
                self.load_timer = QTimer()
                self.load_timer.setSingleShot(True)
                self.load_timer.timeout.connect(self.load_thumbnail)
            # Random delay to stagger loads
            import random
            delay = random.randint(100, 300)
            self.load_timer.start(delay)
    
    def load_thumbnail(self):
        """Load thumbnail image asynchronously."""
        if self.has_loaded:
            return
        self.has_loaded = True
        # Get global async loader
        loader = get_async_loader()
        
        # First check if image is already in cache
        cached_result = loader.get_cached_image(self.image_path, self.thumbnail_size)
        if cached_result:
            # Image is already cached, use it immediately without async loading
            self.on_image_loaded(cached_result)
            return
        
        # Not in cache, proceed with async loading
        # Connect to the loader's signal and store the connection
        self.loader_connection = loader.image_loaded.connect(self.on_image_loaded)
        
        # Queue the image for loading with lower priority
        loader.load_image(self.image_path, self.thumbnail_size, priority=2)
    
    def on_image_loaded(self, result: ImageLoadResult):
        """Handle async image loading completion."""
        # Only process results for this specific image
        if result.image_path != self.image_path:
            return
        
        # Disconnect from the loader signal now that we have our image
        if self.loader_connection:
            loader = get_async_loader()
            loader.image_loaded.disconnect(self.loader_connection)
            self.loader_connection = None
        
        if result.success and result.pixmap:
            # Successfully loaded image - async loader already scaled it
            self.setPixmap(result.pixmap)
            filename = os.path.basename(self.image_path)
            self.setToolTip(f"{filename}")
        else:
            # Failed to load image
            self.setText("⚠️")
            self.setStyleSheet(self.styleSheet() + """
                QLabel { 
                    color: red; 
                    font-size: 10px;
                }
            """)
    
    def set_current(self, is_current: bool):
        """Set whether this is the currently displayed image."""
        self.is_current = is_current
        if is_current:
            self.setStyleSheet("""
                QLabel {
                    border: 3px solid #007acc;
                    border-radius: 6px;
                    background-color: #e6f3ff;
                    padding: 2px;
                }
                QLabel:hover {
                    border-color: #005599;
                    background-color: #cce7ff;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #ddd;
                    border-radius: 6px;
                    background-color: #fafafa;
                    padding: 2px;
                }
                QLabel:hover {
                    border-color: #007acc;
                    background-color: #f0f8ff;
                }
            """)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class SelectedThumbnail(QWidget):
    """Thumbnail widget for selected images with remove button."""
    
    remove_clicked = Signal(str)  # image_path
    
    def __init__(
        self,
        image_path: str,
        size: int = 80,
        show_filename: bool = True,
        compact: bool = False,
    ):
        super().__init__()
        self.image_path = image_path
        self.thumbnail_size = size
        self.show_filename = show_filename
        self.compact = compact
        self.loader_connection = None  # Track the signal connection
        self.has_loaded = False
        self.load_timer = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the thumbnail UI."""
        layout = QVBoxLayout(self)
        if self.compact:
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)
        else:
            layout.setContentsMargins(5, 5, 5, 5)
            layout.setSpacing(3)
        
        # Container for image and remove button
        image_container = QWidget()
        container_padding = 6 if self.compact else 10
        image_container.setFixedSize(
            self.thumbnail_size + container_padding,
            self.thumbnail_size + container_padding,
        )
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image widget
        self.image_widget = QLabel()
        self.image_widget.setFixedSize(self.thumbnail_size, self.thumbnail_size)
        self.image_widget.setAlignment(Qt.AlignCenter)
        padding = 1 if self.compact else 2
        border_radius = 3 if self.compact else 4
        self.image_widget.setStyleSheet(f"""
            QLabel {{
                border: 1px solid #ddd;
                border-radius: {border_radius}px;
                background-color: #fafafa;
                padding: {padding}px;
            }}
        """)
        image_layout.addWidget(self.image_widget)
        
        # Remove button overlay
        self.remove_button = QPushButton("×")
        button_size = 16 if self.compact else 20
        font_size = 10 if self.compact else 12
        self.remove_button.setFixedSize(button_size, button_size)
        self.remove_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: {button_size // 2}px;
                font-size: {font_size}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #cc0000;
            }}
        """)
        self.remove_button.clicked.connect(
            lambda: self.remove_clicked.emit(self.image_path)
        )
        
        # Position remove button in top-right corner
        self.remove_button.setParent(image_container)
        offset_x = max(0, self.thumbnail_size - button_size + 2)
        self.remove_button.move(offset_x, 2)
        self.remove_button.raise_()
        
        layout.addWidget(image_container)
        
        # Filename label
        filename = os.path.basename(self.image_path)
        if self.show_filename:
            display_name = filename
            if len(display_name) > 12:
                display_name = display_name[:9] + "..."
            label_size = 9 if self.compact else 10
            filename_label = QLabel(display_name)
            filename_label.setAlignment(Qt.AlignCenter)
            filename_label.setStyleSheet(f"font-size: {label_size}px; color: #666;")
            filename_label.setToolTip(filename)
            layout.addWidget(filename_label)
        else:
            self.setToolTip(filename)
            self.image_widget.setToolTip(filename)
        
        # Defer thumbnail loading
        self.defer_load_thumbnail()
    
    def cleanup(self):
        """Cleanup signal connections."""
        if hasattr(self, 'load_timer') and self.load_timer:
            self.load_timer.stop()
            self.load_timer = None
        
        if hasattr(self, 'loader_connection') and self.loader_connection:
            try:
                loader = get_async_loader()
                loader.image_loaded.disconnect(self.loader_connection)
                self.loader_connection = None
            except:
                pass
    
    def defer_load_thumbnail(self):
        """Defer thumbnail loading to prevent UI blocking."""
        if not self.has_loaded:
            if not self.load_timer:
                self.load_timer = QTimer()
                self.load_timer.setSingleShot(True)
                self.load_timer.timeout.connect(self.load_thumbnail)
            # Random delay to stagger loads
            import random
            delay = random.randint(150, 350)
            self.load_timer.start(delay)
    
    def load_thumbnail(self):
        """Load thumbnail image."""
        if self.has_loaded:
            return
        self.has_loaded = True
        
        self.image_widget.setText("📷")
        
        # Get global async loader
        loader = get_async_loader()
        
        # First check if image is already in cache
        cached_result = loader.get_cached_image(self.image_path, self.thumbnail_size)
        if cached_result:
            # Image is already cached, use it immediately without async loading
            self.on_image_loaded(cached_result)
            return
        
        # Not in cache, proceed with async loading
        # Connect to the loader's signal and store the connection
        self.loader_connection = loader.image_loaded.connect(self.on_image_loaded)
        loader.load_image(self.image_path, self.thumbnail_size, priority=3)
    
    def on_image_loaded(self, result: ImageLoadResult):
        """Handle async image loading completion."""
        if result.image_path != self.image_path:
            return
        
        # Disconnect from the loader signal now that we have our image
        if self.loader_connection:
            loader = get_async_loader()
            loader.image_loaded.disconnect(self.loader_connection)
            self.loader_connection = None
        
        if result.success and result.pixmap:
            # Async loader already scaled the pixmap
            self.image_widget.setPixmap(result.pixmap)
        else:
            self.image_widget.setText("⚠️")
            self.image_widget.setStyleSheet(
                self.image_widget.styleSheet() + "color: red;"
            )
