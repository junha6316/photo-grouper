from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QLabel, QProgressBar,
    QMessageBox, QSplitter, QStackedWidget, QFrame, QComboBox, QToolButton, QMenu
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction
import os
from typing import List, Dict
import numpy as np

from core.scanner import ImageScanner

from core.grouper import PhotoGrouper
from ui.views import AllPhotosView, GroupedPhotosView, GroupDetailView, SelectedTray

class ProcessingThread(QThread):
    """Thread for background image processing."""
    
    progress_updated = Signal(int, str)  # progress percentage, status message
    processing_finished = Signal(list, dict, list)  # list of groups, embeddings dict, similarities
    images_scanned = Signal(list)  # Emit scanned images immediately for display
    
    def __init__(self, folder_path: str, threshold: float):
        super().__init__()
        self.folder_path = folder_path
        self.threshold = threshold
        
    def run(self):
        try:
            # Step 1: Scan images
            self.progress_updated.emit(10, "Scanning for images...")
            scanner = ImageScanner()
            image_paths = scanner.scan_images(self.folder_path)
            image_paths.sort()
            if not image_paths:
                self.progress_updated.emit(100, "No images found")
                self.processing_finished.emit([], {}, [])
                return
            
            # Emit scanned images immediately so UI can start displaying them
            self.images_scanned.emit(image_paths)
            self.progress_updated.emit(12, f"Found {len(image_paths)} images, extracting features...")
            
            # Step 2: Initialize embedder and fit PCA
            self.progress_updated.emit(15, "Initializing embedder...")
            from core.embedder import ImageEmbedder
            embedder = ImageEmbedder()
            
            def progress_callback(current_idx: int, total: int, current_path: str, eta_seconds=None):
                progress = 20 + int(60 * current_idx / total)  # 20-80%
                message = f"Extracting features {current_idx+1}/{total}"
                if eta_seconds is not None and eta_seconds > 0:
                    eta_min, eta_sec = divmod(int(eta_seconds), 60)
                    if eta_min > 0:
                        message += f" (ETA: {eta_min}m {eta_sec}s)"
                    else:
                        message += f" (Estimated time: {eta_sec}s)"
                self.progress_updated.emit(progress, message)
            
            self.progress_updated.emit(20, "Fitting PCA and extracting features...")
            
            # fit_pca now returns the embeddings dictionary directly
            embeddings = embedder.fit_pca(image_paths, progress_callback=progress_callback)
            
            # Show cache stats
            cache_stats = embedder.get_cache_stats()
            print(f"Cache stats: {cache_stats}")
            
            # Step 5: Group by similarity (always include singles for display control)
            self.progress_updated.emit(90, "Grouping similar images...")
            grouper = PhotoGrouper()
            groups, similarities = grouper.group_by_threshold(embeddings, self.threshold, min_group_size=1, strict_mode="min")
            
            # Step 6: Sort clusters by inter-cluster similarity (if enabled)
            
            self.progress_updated.emit(95, "Sorting clusters by similarity...")
            sorted_groups, sorted_similarities = grouper.sort_clusters_by_similarity(groups, embeddings, similarities, self.threshold)
            
            # Show which method was used
            method = "Direct similarity algorithm"
            print(f"Grouping completed using {method} with {len(embeddings)} images")
            print(f"Clusters sorted by inter-cluster similarity: {len(sorted_groups)} groups")
            
            self.progress_updated.emit(100, f"Found {len(sorted_groups)} groups")
            self.processing_finished.emit(sorted_groups, embeddings, sorted_similarities)
            
        except Exception as e:
            print(f"DEBUG: Error: {str(e)}")
            self.progress_updated.emit(100, f"Error: {str(e)}")
            self.processing_finished.emit([], {}, [])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Grouper")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data
        self.current_folder = ""
        self.current_groups = []
        self.current_embeddings = {}
        self.all_image_paths = []  # Store all scanned images
        
        # Global selection state - tracks selected images across all clusters
        self.global_selected_images = set()
        
        # Processing thread
        self.processing_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Top controls
        controls_layout = QHBoxLayout()

        # Folder selection
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        controls_layout.addWidget(self.folder_button)

        self.folder_label = QLabel("No folder selected")
        controls_layout.addWidget(self.folder_label)

        controls_layout.addStretch()

        # Preset menu for grouping tightness
        from PySide6.QtWidgets import QSpinBox, QCheckBox
        controls_layout.addWidget(QLabel("Grouping:"))

        self.grouping_button = QToolButton()
        self.grouping_button.setPopupMode(QToolButton.InstantPopup)
        self.grouping_menu = QMenu(self.grouping_button)
        self.grouping_preset_actions = {}

        for label, threshold in (("Strict", 0.90), ("Normal", 0.85), ("Loose", 0.75)):
            action = QAction(f"{label} ({threshold:.2f})", self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, t=threshold: self.apply_preset(t))
            self.grouping_menu.addAction(action)
            self.grouping_preset_actions[label.lower()] = action

        self.grouping_menu.addSeparator()

        self.grouping_advanced_action = QAction("Advanced settings", self)
        self.grouping_advanced_action.setCheckable(True)
        self.grouping_advanced_action.setChecked(False)
        self.grouping_advanced_action.triggered.connect(self.toggle_advanced_settings)
        self.grouping_menu.addAction(self.grouping_advanced_action)

        self.grouping_button.setMenu(self.grouping_menu)
        self.grouping_button.setText("Normal")
        self.grouping_button.setStyleSheet("""
            QToolButton {
                padding: 3px 18px 3px 8px;
                font-size: 11px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f8f8;
            }
            QToolButton:hover:enabled {
                background-color: #e8e8e8;
            }
            QToolButton::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                right: 6px;
                width: 8px;
                height: 8px;
            }
        """)
        controls_layout.addWidget(self.grouping_button)

        layout.addLayout(controls_layout)

        # Advanced settings panel (initially hidden)
        self.advanced_panel = QFrame()
        self.advanced_panel.setFrameShape(QFrame.StyledPanel)
        self.advanced_panel.setVisible(False)
        advanced_layout = QVBoxLayout(self.advanced_panel)

        # Threshold slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Similarity Threshold:"))

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(50)  # 0.50
        self.threshold_slider.setMaximum(99)  # 0.99
        self.threshold_slider.setValue(85)    # 0.85 default
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        slider_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel("0.85")
        self.threshold_label.setMinimumWidth(40)
        slider_layout.addWidget(self.threshold_label)
        advanced_layout.addLayout(slider_layout)

        # Other advanced controls
        other_controls_layout = QHBoxLayout()

        # Minimum group size
        other_controls_layout.addWidget(QLabel("Min group size:"))
        self.min_group_spinbox = QSpinBox()
        self.min_group_spinbox.setMinimum(1)
        self.min_group_spinbox.setMaximum(10)
        self.min_group_spinbox.setValue(2)  # Default to groups with 2+ images
        self.min_group_spinbox.valueChanged.connect(self.on_min_group_changed)
        other_controls_layout.addWidget(self.min_group_spinbox)

        other_controls_layout.addStretch()

        # Cluster similarity sorting toggle
        self.sort_clusters_checkbox = QCheckBox("Sort similar clusters together")
        self.sort_clusters_checkbox.setChecked(True)  # Default enabled
        self.sort_clusters_checkbox.setToolTip("Group similar clusters together for better visualization")
        self.sort_clusters_checkbox.stateChanged.connect(self.on_cluster_sort_changed)
        other_controls_layout.addWidget(self.sort_clusters_checkbox)

        advanced_layout.addLayout(other_controls_layout)
        layout.addWidget(self.advanced_panel)

        self._update_grouping_preset_ui(self.threshold_slider.value() / 100.0)
        
        # Progress bar
        from PySide6.QtWidgets import QSizePolicy
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.status_label)
        
        # View mode selector
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View:"))

        self.view_combo = QComboBox()
        self.view_combo.addItem("Grouped", "grouped")
        self.view_combo.addItem("All Photos", "all")
        self.view_combo.currentIndexChanged.connect(self.on_view_combo_changed)
        view_layout.addWidget(self.view_combo)

        view_layout.addStretch()
        layout.addLayout(view_layout)

        # Main workspace views
        self.all_photos_view = AllPhotosView(self)
        self.grouped_photos_view = GroupedPhotosView(self)
        self.group_detail_view = GroupDetailView(self)
        self.selected_tray = SelectedTray(self)

        # Connect selection signals for synchronization
        self.all_photos_view.selection_changed.connect(self.on_selection_changed_from_view)
        self.group_detail_view.selection_changed.connect(self.on_selection_changed_from_view)
        self.selected_tray.selection_changed.connect(self.on_selection_changed_from_view)

        # Connect group click to detail view
        self.grouped_photos_view.group_clicked.connect(self.on_group_clicked)

        # Selected tray actions
        self.selected_tray.clear_requested.connect(self.clear_global_selection)
        self.selected_tray.review_requested.connect(self.open_selected_review)
        self.selected_tray.export_requested.connect(self.export_selected_images)

        # Center stack (all vs group detail)
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self.all_photos_view)
        self.center_stack.addWidget(self.group_detail_view)

        # Layout with resizable panels
        self.content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.addWidget(self.grouped_photos_view)
        self.content_splitter.addWidget(self.center_stack)
        self.content_splitter.addWidget(self.selected_tray)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 3)
        self.content_splitter.setStretchFactor(2, 1)

        self.grouped_photos_view.setMinimumWidth(300)
        self.selected_tray.setMinimumWidth(260)
        # Initial layout: 25% left panel, 75% center (selected tray hidden initially)
        self.content_splitter.setSizes([360, 900, 0])

        layout.addWidget(self.content_splitter, 1)

        # Keep reference to preview panel for compatibility
        self.preview_panel = self.grouped_photos_view.get_preview_panel()

        self.view_mode = ""
        self.set_view_mode("grouped")
        
    def select_folder(self):
        """Open folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if folder:
            self.current_folder = folder
            self.folder_label.setText(f"Folder: {os.path.basename(folder)}")
            self.process_folder()
    
    def process_folder(self):
        """Start processing the selected folder."""
        if not self.current_folder:
            return
            
        # Clear previous data
        self.all_photos_view.clear()
        self.grouped_photos_view.clear()
        self.group_detail_view.clear()
        self.selected_tray.sync_selected_images(self.global_selected_images)
        self.grouped_photos_view.set_summary_text("")
        self.grouped_photos_view.sync_selected_counts(self.global_selected_images)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.folder_button.setEnabled(False)
        
        # Start processing thread
        threshold = self.threshold_slider.value() / 100.0
        self.processing_thread = ProcessingThread(self.current_folder, threshold)
        self.processing_thread.progress_updated.connect(self.on_progress_updated)
        self.processing_thread.images_scanned.connect(self.on_images_scanned)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.start()
    
    def on_progress_updated(self, percentage: int, message: str):
        """Handle progress updates from processing thread."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        
        # Update grouped view with progress if it's showing processing message
        if hasattr(self.grouped_photos_view, 'update_processing_progress'):
            self.grouped_photos_view.update_processing_progress(percentage, message)
    
    def on_images_scanned(self, image_paths: List[str]):
        """Handle immediate display of scanned images before processing."""
        # Store the scanned images
        self.all_image_paths = image_paths
        
        # Immediately populate the All Photos view
        self.all_photos_view.set_images(image_paths)
        self.set_view_mode("all")

        # Keep grouped view showing processing with initial status
        self.grouped_photos_view.show_processing_message(
            f"Found {len(image_paths)} images\nExtracting image features..."
        )
    
    def on_processing_finished(self, groups: List[List[str]], embeddings: Dict[str, np.ndarray], similarities: List[float]):
        """Handle completion of processing."""
        self.current_groups = groups
        self.current_embeddings = embeddings
        self.current_similarities = similarities
        self.progress_bar.setVisible(False)
        self.folder_button.setEnabled(True)
        
        # Extract all image paths from groups
        self.all_image_paths = []
        for group in groups:
            self.all_image_paths.extend(group)
        self.all_image_paths.sort()
        
        if groups:
            min_display_size = self.min_group_spinbox.value()
            
            # Update all views (All photos already set in on_images_scanned)
            # Just update the grouped view with the final groups
            self.grouped_photos_view.set_groups(groups, min_display_size, similarities)
            self.group_detail_view.clear()
            self.selected_tray.sync_selected_images(self.global_selected_images)
            
            total_images = sum(len(g) for g in groups)
            
            # Calculate actual displayed group count (accounting for singles consolidation)
            displayed_count = self._calculate_displayed_group_count(groups, min_display_size)
            summary_text = f"Found {displayed_count} groups with {total_images} total images"
            self.grouped_photos_view.set_summary_text(summary_text)
            self.grouped_photos_view.sync_selected_counts(self.global_selected_images)
            
            # Enable deduplication button if we have groups suitable for deduplication
            dedup_groups = [g for g in groups if len(g) > 1]
            # self.dedup_button.setEnabled(len(dedup_groups) > 0)
            
            # Get cache stats for display
            try:
                # Get a sample embedder to check cache stats
                from core.embedder import ImageEmbedder
                temp_embedder = ImageEmbedder()
                cache_stats = temp_embedder.get_cache_stats()
                cached_count = cache_stats.get('total_embeddings', 0)
            except Exception:
                cached_count = 0
            
            status_parts = []
            if cached_count:
                status_parts.append(f"{cached_count} cached embeddings")
            if dedup_groups:
                status_parts.append(f"{len(dedup_groups)} groups ready for deduplication")
            self.status_label.setText(" | ".join(status_parts))
        else:
            self.all_photos_view.clear()
            self.grouped_photos_view.clear()
            self.group_detail_view.clear()
            self.selected_tray.sync_selected_images(self.global_selected_images)
            # self.dedup_button.setEnabled(False)
            self.grouped_photos_view.set_summary_text("No similar images found")
            self.status_label.setText("No similar images found or processing failed")
            self.grouped_photos_view.sync_selected_counts(self.global_selected_images)
    
    def apply_preset(self, threshold: float):
        """Apply a preset threshold value."""
        # Update slider to match preset (will trigger on_threshold_changed)
        slider_value = int(threshold * 100)
        self.threshold_slider.setValue(slider_value)

    def _update_grouping_preset_ui(self, threshold: float):
        preset_name = None
        if threshold == 0.90:
            preset_name = "strict"
            label = "Strict"
        elif threshold == 0.85:
            preset_name = "normal"
            label = "Normal"
        elif threshold == 0.75:
            preset_name = "loose"
            label = "Loose"
        else:
            label = f"Custom {threshold:.2f}"

        for action in self.grouping_preset_actions.values():
            action.setChecked(False)

        if preset_name:
            self.grouping_preset_actions[preset_name].setChecked(True)

        self.grouping_button.setText(label)

    def toggle_advanced_settings(self, checked: bool = None):
        """Toggle visibility of advanced settings panel."""
        if checked is None:
            new_visible = not self.advanced_panel.isVisible()
        else:
            new_visible = checked

        self.advanced_panel.setVisible(new_visible)
        if hasattr(self, "grouping_advanced_action"):
            self.grouping_advanced_action.blockSignals(True)
            self.grouping_advanced_action.setChecked(new_visible)
            self.grouping_advanced_action.blockSignals(False)

    def on_threshold_changed(self):
        """Handle threshold slider changes."""
        threshold = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        self._update_grouping_preset_ui(threshold)

        # If we have current data, reprocess with new threshold
        if self.current_folder and hasattr(self, 'current_embeddings') and self.current_embeddings:
            self.regroup_with_threshold(threshold)
    
    def regroup_with_threshold(self, threshold: float):
        """Regroup existing embeddings with new threshold."""
        if not self.current_embeddings:
            return
            
        # Quick regrouping without re-extracting embeddings
        try:
            grouper = PhotoGrouper()
            groups, similarities = grouper.group_by_threshold(self.current_embeddings, threshold, min_group_size=1)
            sorted_groups, sorted_similarities = grouper.sort_clusters_by_similarity(groups, self.current_embeddings, similarities, threshold)
            
            
            # Debug output to console
            print(f"Threshold: {threshold}, Groups: {len(sorted_groups)}, Total images: {sum(len(g) for g in sorted_groups)}")
            print(f"Group sizes: {[len(g) for g in sorted_groups]}")
            
            self.current_groups = sorted_groups
            self.current_similarities = sorted_similarities
            min_display_size = self.min_group_spinbox.value()
            
            # Update grouped photos view
            self.grouped_photos_view.set_groups(sorted_groups, min_display_size, sorted_similarities)
            self.group_detail_view.clear()
            
            # Update deduplication button availability
            dedup_groups = [g for g in sorted_groups if len(g) > 1]
            # self.dedup_button.setEnabled(len(dedup_groups) > 0)
            
            # Calculate actual displayed group count (accounting for singles consolidation)
            displayed_count = self._calculate_displayed_group_count(sorted_groups, min_display_size)
            total_images = sum(len(g) for g in sorted_groups)
            summary_text = f"Found {displayed_count} groups with {total_images} total images"
            self.grouped_photos_view.set_summary_text(summary_text)
            self.grouped_photos_view.sync_selected_counts(self.global_selected_images)

            status_parts = [f"Regrouped at {threshold:.2f}"]
            if dedup_groups:
                status_parts.append(f"{len(dedup_groups)} groups ready for deduplication")
            self.status_label.setText(" | ".join(status_parts))
        except Exception as e:
            print(f"Regrouping error: {str(e)}")
            self.status_label.setText(f"Regrouping failed: {str(e)}")
    
    def on_min_group_changed(self):
        """Handle minimum group size changes."""
        # Re-process with new minimum group size
        if hasattr(self, 'current_embeddings') and self.current_embeddings:
            threshold = self.threshold_slider.value() / 100.0
            self.regroup_with_threshold(threshold)
    
    def on_cluster_sort_changed(self):
        """Handle cluster similarity sorting toggle."""
        # Re-process with current settings to apply/remove sorting
        if hasattr(self, 'current_embeddings') and self.current_embeddings:
            threshold = self.threshold_slider.value() / 100.0
            self.regroup_with_threshold(threshold)
    
    def _update_selected_tray_visibility(self):
        """Show or hide the selected tray based on whether images are selected."""
        has_selections = len(self.global_selected_images) > 0

        # Get current splitter sizes
        sizes = self.content_splitter.sizes()

        if has_selections and sizes[2] == 0:
            # Show the selected tray by giving it space
            total_width = sum(sizes)
            # Give 20% to selected tray, adjust center panel
            new_tray_width = int(total_width * 0.2)
            new_center_width = sizes[1] - new_tray_width
            self.content_splitter.setSizes([sizes[0], new_center_width, new_tray_width])
        elif not has_selections and sizes[2] > 0:
            # Hide the selected tray by collapsing it
            # Give its space back to center panel
            self.content_splitter.setSizes([sizes[0], sizes[1] + sizes[2], 0])

    def _calculate_displayed_group_count(self, groups: list, min_display_size: int) -> int:
        """
        Calculate the actual number of groups that will be displayed to the user.
        This accounts for singles consolidation and min_display_size filtering.
        """
        displayed_count = 0
        has_singles = False
        
        for group in groups:
            if len(group) >= min_display_size:
                displayed_count += 1
            elif len(group) > 5:  # Likely the singles group
                has_singles = True
        
        # Add 1 for the singles group if it exists
        if has_singles:
            displayed_count += 1
            
        return displayed_count

    def on_view_combo_changed(self, index: int):
        """Handle view selector changes."""
        mode = self.view_combo.itemData(index)
        if mode:
            self.set_view_mode(mode)

    def set_view_mode(self, mode: str):
        """Switch between grouped detail and all photos views."""
        if mode == self.view_mode:
            return

        self.view_mode = mode
        if mode == "grouped":
            self.center_stack.setCurrentWidget(self.group_detail_view)
        else:
            self.center_stack.setCurrentWidget(self.all_photos_view)

        index = self.view_combo.findData(mode)
        if index >= 0 and index != self.view_combo.currentIndex():
            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentIndex(index)
            self.view_combo.blockSignals(False)

    def on_group_clicked(self, images: List[str], group_number: int, is_singles_group: bool, similarity):
        """Open a group in the detail view."""
        self.group_detail_view.set_cluster(images, group_number, is_singles_group, similarity)
        self.set_view_mode("grouped")

    def open_selected_review(self):
        """Open a review dialog for selected images."""
        if not self.global_selected_images:
            return

        from ui.selected_view_dialog import SelectedViewDialog
        dialog = SelectedViewDialog(list(self.global_selected_images), self)
        dialog.exec()

    def export_selected_images(self):
        """Open export dialog for selected images."""
        if not self.global_selected_images:
            return

        from ui.export_dialog import ExportDialog
        dialog = ExportDialog(list(self.global_selected_images), self)
        dialog.exec()

    def update_selected_tray(self):
        """Update the selected tray with current selections."""
        self.selected_tray.sync_selected_images(self.global_selected_images)
        self._update_selected_tray_visibility()
    
    def get_global_selected_images(self):
        """Get the set of globally selected image paths."""
        return self.global_selected_images.copy()
    
    def clear_global_selection(self):
        """Clear all global selections."""
        self.global_selected_images.clear()
        self.all_photos_view.clear_selection()
        self.group_detail_view.clear_selection()
        self.selected_tray.clear_selection()
        self.grouped_photos_view.sync_selected_counts(self.global_selected_images)
    
    def add_to_global_selection(self, image_path: str):
        """Add an image to global selection."""
        if image_path in self.global_selected_images:
            return
        self.global_selected_images.add(image_path)
        self.all_photos_view.sync_selection(image_path, True)
        self.group_detail_view.sync_selection(image_path, True)
        self.update_selected_tray()
    
    def remove_from_global_selection(self, image_path: str):
        """Remove an image from global selection."""
        if image_path not in self.global_selected_images:
            return
        self.global_selected_images.discard(image_path)
        self.all_photos_view.sync_selection(image_path, False)
        self.group_detail_view.sync_selection(image_path, False)
        self.update_selected_tray()
    
    def on_selection_changed_from_view(self, image_path: str, is_selected: bool):
        """Handle selection changes from any view."""
        # Update global selection state
        if is_selected:
            self.add_to_global_selection(image_path)
        else:
            self.remove_from_global_selection(image_path)
        self.grouped_photos_view.sync_selected_counts(self.global_selected_images)
    
    def start_deduplication_session(self):
        """Start a new deduplication session."""
        if not self.current_groups or not self.current_embeddings:
            QMessageBox.warning(self, "No Data", "Please process a folder first to start deduplication.")
            return
        
        try:
            # Import and launch the deduplication dialog
            from ui.deduplication_dialog import DeduplicationDialog
            
            # Create and show the deduplication dialog
            dedup_dialog = DeduplicationDialog(self.current_embeddings, self.current_groups, self)
            dedup_dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start deduplication session: {str(e)}")
    
    def on_session_ready(self, session):
        """Handle when a session is ready to start."""
        try:
            from ui.select_session_dialog import SelectSessionDialog
            
            # Create and show the session dialog
            session_dialog = SelectSessionDialog(session, self)
            session_dialog.session_completed.connect(self.on_session_completed)
            session_dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open session dialog: {str(e)}")
    
    def on_session_completed(self, results):
        """Handle session completion."""
        try:
            winners = results.get('winners', [])
            total_groups = results.get('total_groups', 0)
            completed_groups = results.get('completed_groups', 0)
            skipped_groups = results.get('skipped_groups', 0)
            
            # Show results summary
            msg = QMessageBox(self)
            msg.setWindowTitle("Deduplication Complete")
            msg.setIcon(QMessageBox.Information)
            msg.setText("Deduplication session completed successfully!")
            
            details = (
                f"Session Results:\n"
                f"• Total groups processed: {total_groups}\n"
                f"• Completed groups: {completed_groups}\n"
                f"• Skipped groups: {skipped_groups}\n"
                f"• Winners selected: {len(winners)}\n\n"
                f"Winners List:\n"
            )
            
            for winner_info in winners[:10]:  # Show first 10 winners
                winner_file = winner_info.get('winner', '')
                eliminated_count = len(winner_info.get('eliminated', []))
                filename = os.path.basename(winner_file)
                details += f"• {filename} (eliminated {eliminated_count} duplicates)\n"
            
            if len(winners) > 10:
                details += f"... and {len(winners) - 10} more winners\n"
            
            msg.setDetailedText(details)
            msg.exec()
            
            # Update status
            self.status_label.setText(f"Deduplication complete: {len(winners)} winners from {completed_groups} groups")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error processing session results: {str(e)}")
