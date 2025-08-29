from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from typing import List
import os

from ui.utils.async_image_loader import get_async_loader, ImageLoadResult
from ui.utils.image_preloader import ViewportImagePreloader
from ui.components.layouts import FlowLayout

class PreviewThumbnailWidget(QLabel):
    """Widget to display a single thumbnail image with async loading in preview panel."""
    
    def __init__(self, image_path: str, size: int = 150, parent_group_widget=None):
        super().__init__()
        self.image_path = image_path
        self.thumbnail_size = size
        self.is_loading = True
        self.parent_group_widget = parent_group_widget
        self.loader_connection = None  # Track the signal connection
        
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setContentsMargins(0, 0, 0, 0)
        self.setScaledContents(True)
        
        self._update_style()
        
        # Show loading indicator
        self.show_loading()
        
        # Start async loading
        self.load_thumbnail_async()
    
    def __del__(self):
        """Cleanup signal connections when widget is destroyed."""
        if hasattr(self, 'loader_connection') and self.loader_connection:
            try:
                loader = get_async_loader()
                loader.image_loaded.disconnect(self.loader_connection)
            except:
                pass  # Ignore errors if loader is already cleaned up
    
    def _update_style(self):
        """Update widget style."""
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
                padding: 0px;
                margin: 0px;
            }
        """)
    
    
    def show_loading(self):
        """Show loading indicator."""
        self.setText("📷")
        self.setStyleSheet(self.styleSheet() + """
            QLabel { 
                color: #ccc; 
                font-size: 24px;
            }
        """)
    
    def load_thumbnail_async(self):
        """Start async thumbnail loading."""
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
        loader.load_image(self.image_path, self.thumbnail_size, priority=0)  # Lower priority than cluster dialog
    
    def on_image_loaded(self, result: ImageLoadResult):
        """Handle async image loading completion."""
        if result.image_path != self.image_path:
            return
        
        # Disconnect from the loader signal now that we have our image
        if self.loader_connection:
            loader = get_async_loader()
            loader.image_loaded.disconnect(self.loader_connection)
            self.loader_connection = None
        
        self.is_loading = False
        
        if result.success and result.pixmap:
            self.setPixmap(result.pixmap)
            self.setToolTip(f"{os.path.basename(self.image_path)}\n{self.image_path}")
            # Update style to reflect current selection state
            self._update_style()
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
    
    

class GroupWidget(QFrame):
    """Widget to display a group of similar images."""
    
    def __init__(self, group_images: List[str], group_number: int, is_singles_group: bool = False, similarity: float = None):
        super().__init__()
        self.group_images = group_images
        self.group_number = group_number
        self.is_singles_group = is_singles_group
        self.similarity = similarity
        self.thumbnails = []  # Keep track of thumbnail widgets
        
        # Make the widget clickable
        self.setCursor(Qt.PointingHandCursor)
        
        self.setFrameStyle(QFrame.Box)
        # Set object name for specific CSS targeting
        self.setObjectName("GroupWidget")
        self.setStyleSheet("""
            QFrame#GroupWidget {
                border: 1px solid #ccc;
                border-radius: 6px;
                margin: 3px;
                background-color: white;
            }
            QFrame#GroupWidget:hover {
                border-color: #007acc;
                background-color: #f0f8ff;
            }
        """)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the group UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Group header - more compact
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(11)
        
        # Header with selection controls
        header_layout = QHBoxLayout()
        
        if self.is_singles_group:
            header_text = f"Single Images ({len(self.group_images)} images)"
            header_color = "#666"
            header_style = "font-style: italic;"
        else:
            # Include similarity score if available
            if self.similarity is not None and self.similarity > 0:
                header_text = f"Group {self.group_number} (Similarity: {self.similarity:.2f}) - {len(self.group_images)} images"
            else:
                header_text = f"Group {self.group_number} ({len(self.group_images)} images)"
            header_color = "#333"
            header_style = ""
        
        header = QLabel(header_text)
        header.setFont(header_font)
        header.setStyleSheet(f"color: {header_color}; padding: 2px; {header_style}")
        header_layout.addWidget(header)
        
        layout.addLayout(header_layout)
        
        # Thumbnails using responsive flow layout
        thumbnails_widget = QWidget()
        flow_layout = FlowLayout(thumbnails_widget)
        flow_layout.setSpacing(5)
        flow_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create compact thumbnails
        display_count = min(8, len(self.group_images))  # Show max 8 thumbnails
        for image_path in self.group_images[:display_count]:
            thumbnail = PreviewThumbnailWidget(image_path, size=150, parent_group_widget=self)  # Match grid view size
            self.thumbnails.append(thumbnail)
            flow_layout.addWidget(thumbnail)
        
        # Show count if there are more images
        if len(self.group_images) > display_count:
            more_card = QLabel(f"+ {len(self.group_images) - display_count} more")
            more_card.setFixedSize(150, 150)  # Same size as thumbnails
            more_card.setAlignment(Qt.AlignCenter)
            # more_card.setCursor(Qt.PointingHandCursor)
            more_card.setStyleSheet("""
                QLabel {
                    color: #666; 
                    font-weight: bold;
                    font-size: 12px;
                    border: 2px dashed #ccc;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                }
               
            """)
            flow_layout.addWidget(more_card)
        
        layout.addWidget(thumbnails_widget)
    
    
    def mousePressEvent(self, event):
        """Handle mouse clicks on the group widget."""
        if event.button() == Qt.LeftButton:
            print(f"GroupWidget clicked: Group {self.group_number} with {len(self.group_images)} images")
            self.show_cluster_details()
        super().mousePressEvent(event)
    
    def show_cluster_details(self):
        """Show detailed view of this cluster."""
        try:
            from ui.cluster_dialog import ClusterDialog
            # Find the main window by traversing up the parent hierarchy
            main_window = self
            while main_window:
                if hasattr(main_window, 'add_to_global_selection'):
                    print(f"Found main window with add_to_global_selection: {type(main_window)}")
                    break
                main_window = main_window.parent()
            
            # If we couldn't find the main window with add_to_global_selection,
            # try to find QMainWindow instance
            if not main_window or not hasattr(main_window, 'add_to_global_selection'):
                from PySide6.QtWidgets import QMainWindow
                main_window = self
                while main_window:
                    if isinstance(main_window, QMainWindow):
                        print(f"Found QMainWindow: {type(main_window)}")
                        break
                    main_window = main_window.parent()
            
            print(f"Creating ClusterDialog with main_window: {type(main_window) if main_window else 'None'}")
            dialog = ClusterDialog(self.group_images, self.group_number, main_window)
            dialog.exec()
        except Exception as e:
            print(f"Error showing cluster details: {e}")
            import traceback
            traceback.print_exc()

class PreviewPanel(QWidget):
    """Main panel to display all photo groups."""
    
    def __init__(self):
        super().__init__()
        self.group_widgets = []
        self.preloader = ViewportImagePreloader(thumbnail_size=150)
        self.viewport_timer = QTimer()
        self.viewport_timer.timeout.connect(self._check_viewport)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the preview panel UI."""
        # Create scroll area for groups
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Content widget for scroll area
        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)
        
        # Default message
        self.default_label = QLabel("Select a folder to start grouping photos")
        self.default_label.setAlignment(Qt.AlignCenter)
        self.default_label.setStyleSheet("color: #888; font-size: 16px; padding: 50px;")
        self.layout.addWidget(self.default_label)
        
        self.scroll_area.setWidget(self.content_widget)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll_area)
        
        # Connect scroll event for viewport detection
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll)
    
    def display_groups(self, groups: List[List[str]], min_display_size: int = 2, similarities: List[float] = None):
        """
        Display photo groups in the panel.
        
        Args:
            groups: List of image groups (includes singles group)
            min_display_size: Minimum group size to display (singles group always shown)
            similarities: List of average similarity scores for each group
        """
        print(f"DEBUG: display_groups called with {len(groups)} groups, min_display_size={min_display_size}")
        for i, group in enumerate(groups):
            print(f"  Group {i}: {len(group)} images")
        
        self.clear()
        
        if not groups:
            no_groups_label = QLabel("No similar photo groups found")
            no_groups_label.setAlignment(Qt.AlignCenter)
            no_groups_label.setStyleSheet("color: #888; font-size: 16px; padding: 50px;")
            self.layout.addWidget(no_groups_label)
            return
        
        # Sort groups by size (largest first)
        sorted_groups = sorted(groups, key=len, reverse=True)
        
        # Separate multi-image groups from potential singles groups
        groups_to_show = []
        potential_singles = []
        
        for group in sorted_groups:
            if len(group) >= min_display_size:
                groups_to_show.append(group)
            else:
                potential_singles.append(group)
        
        # If we have remaining smaller groups, treat the largest one as the singles group
        # This is because the grouper collects all single images into one group
        if potential_singles:
            # Sort potential singles by size (largest first) to get the actual singles group
            potential_singles.sort(key=len, reverse=True)
            singles_group = potential_singles[0]  # Take the largest group of small groups
            groups_to_show.append(singles_group)
        
        print(f"DEBUG: groups_to_show has {len(groups_to_show)} groups")
        
        if not groups_to_show:
            print("DEBUG: No groups to show!")
            no_groups_label = QLabel(f"No groups with {min_display_size}+ images found")
            no_groups_label.setAlignment(Qt.AlignCenter)
            no_groups_label.setStyleSheet("color: #888; font-size: 16px; padding: 50px;")
            self.layout.addWidget(no_groups_label)
            return
        
        # Clear previous group widgets
        self.group_widgets = []
        
        # Add each group
        for i, group in enumerate(groups_to_show, 1):
            # Check if this is the singles group (it's the last one and smaller than min_display_size)
            is_singles = (i == len(groups_to_show) and len(group) < min_display_size)
            
            # Get similarity score for this group
            similarity = None
            if similarities:
                # Find the original index of this group to get its similarity
                original_idx = groups.index(group)
                if original_idx < len(similarities):
                    similarity = similarities[original_idx]
            
            group_widget = GroupWidget(group, i, is_singles_group=is_singles, similarity=similarity)
            self.group_widgets.append(group_widget)
            self.layout.addWidget(group_widget)
        
        # Add stretch to push everything to top
        self.layout.addStretch()
        
        # Start viewport monitoring
        self.viewport_timer.start(500)  # Check every 500ms
        self._check_viewport()  # Initial check
    def clear(self):
        """Clear all widgets from the panel."""
        self.group_widgets = []
        self.preloader.clear_preload_cache()
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _on_scroll(self):
        """Handle scroll events for viewport-based preloading."""
        # Debounce with timer restart
        self.viewport_timer.stop()
        self.viewport_timer.start(100)  # Check after 100ms of no scrolling
    
    def _check_viewport(self):
        """Check which images are in viewport and preload them."""
        if not self.group_widgets:
            return
        
        # Get scroll position and viewport height
        scroll_bar = self.scroll_area.verticalScrollBar()
        scroll_pos = scroll_bar.value()
        viewport_height = self.scroll_area.viewport().height()
        
        visible_images = []
        nearby_images = []
        
        # Simple viewport detection based on widget y-position
        current_y = 0
        for group_widget in self.group_widgets:
            if not group_widget.isVisible():
                continue
            
            widget_height = group_widget.height()
            
            # Check if widget is in viewport
            if current_y + widget_height >= scroll_pos and current_y <= scroll_pos + viewport_height:
                # Widget is visible - high priority
                for img_path in group_widget.group_images[:10]:  # First 10 images
                    visible_images.append(img_path)
            elif abs(current_y - scroll_pos) <= viewport_height * 2:
                # Widget is nearby (within 2 viewport heights) - medium priority
                for img_path in group_widget.group_images[:5]:  # First 5 images
                    nearby_images.append(img_path)
            
            current_y += widget_height + self.layout.spacing()
        
        # Trigger preloading
        if visible_images or nearby_images:
            self.preloader.preload_viewport_images(visible_images, nearby_images)