from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QPushButton, QFileDialog, QSlider, QLabel, QProgressBar,
    QMessageBox, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QShortcut, QKeySequence
import os
from typing import List, Dict
import numpy as np

from core.scanner import ImageScanner
from core.embedder import ImageEmbedder
from core.grouper import PhotoGrouper
from ui.views import AllPhotosView, GroupedPhotosView, SelectedPhotosView

class ProcessingThread(QThread):
    """Thread for background image processing."""
    
    progress_updated = Signal(int, str)  # progress percentage, status message
    processing_finished = Signal(list, dict)  # list of groups, embeddings dict
    
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
                self.processing_finished.emit([], {})
                return
            
            # Step 2: Initialize embedder and fit PCA
            self.progress_updated.emit(15, "Initializing embedder...")
            embedder = ImageEmbedder()
            
            # Step 3: Fit PCA on subset or all images
            self.progress_updated.emit(20, "Fitting PCA...")
            # Use all images for PCA fitting (or subset if too many)
            pca_sample = image_paths[:min(1000, len(image_paths))]
            embedder.fit_pca(pca_sample)
            
            # Step 4: Extract PCA embeddings
            embeddings = {}
            total_images = len(image_paths)
            
            for i, path in enumerate(image_paths):
                progress = 30 + int(50 * i / total_images)  # 30-80%
                self.progress_updated.emit(progress, f"Extracting features {i+1}/{total_images}")
                embeddings[path] = embedder.get_embedding(path)
            
            # Show cache stats
            cache_stats = embedder.get_cache_stats()
            print(f"Cache stats: {cache_stats}")
            
            # Step 5: Group by similarity (always include singles for display control)
            self.progress_updated.emit(90, "Grouping similar images...")
            grouper = PhotoGrouper()
            groups = grouper.group_by_threshold(embeddings, self.threshold, min_group_size=1)
            
            # Step 6: Sort clusters by inter-cluster similarity (if enabled)
            if hasattr(self, 'sort_clusters_checkbox') and self.sort_clusters_checkbox.isChecked():
                self.progress_updated.emit(95, "Sorting clusters by similarity...")
                sorted_groups = grouper.sort_clusters_by_similarity(groups, embeddings, self.threshold)
            else:
                sorted_groups = groups
            
            # Show which method was used
            method = "Direct similarity algorithm"
            print(f"Grouping completed using {method} with {len(embeddings)} images")
            print(f"Clusters sorted by inter-cluster similarity: {len(sorted_groups)} groups")
            
            self.progress_updated.emit(100, f"Found {len(sorted_groups)} groups")
            self.processing_finished.emit(sorted_groups, embeddings)
            
        except Exception as e:
            self.progress_updated.emit(100, f"Error: {str(e)}")
            self.processing_finished.emit([], {})

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
        
        # Minimum group size
        from PySide6.QtWidgets import QSpinBox, QCheckBox
        controls_layout.addWidget(QLabel("Min group size:"))
        self.min_group_spinbox = QSpinBox()
        self.min_group_spinbox.setMinimum(1)
        self.min_group_spinbox.setMaximum(10)
        self.min_group_spinbox.setValue(2)  # Default to groups with 2+ images
        self.min_group_spinbox.valueChanged.connect(self.on_min_group_changed)
        controls_layout.addWidget(self.min_group_spinbox)
        
        # Cluster similarity sorting toggle
        self.sort_clusters_checkbox = QCheckBox("Sort similar clusters together")
        self.sort_clusters_checkbox.setChecked(True)  # Default enabled
        self.sort_clusters_checkbox.setToolTip("Group similar clusters together for better visualization")
        self.sort_clusters_checkbox.stateChanged.connect(self.on_cluster_sort_changed)
        controls_layout.addWidget(self.sort_clusters_checkbox)
        
        # Deduplication Session button
        # self.dedup_button = QPushButton("🎯 Start Deduplication")
        # self.dedup_button.setEnabled(False)
        # self.dedup_button.clicked.connect(self.start_deduplication_session)
        # self.dedup_button.setStyleSheet("""
        #     QPushButton {
        #         padding: 8px 15px;
        #         font-size: 12px;
        #         font-weight: bold;
        #         border: 2px solid #e74c3c;
        #         border-radius: 4px;
        #         background-color: #fadbd8;
        #         color: #c0392b;f
        #         min-width: 140px;
        #     }
        #     QPushButton:hover:enabled {
        #         background-color: #f5b7b1;
        #     }
        #     QPushButton:disabled {
        #         color: #999;
        #         background-color: #f5f5f5;
        #         border-color: #ddd;
        #     }
        # """)
        # self.dedup_button.setToolTip("Start a champion vs challenger deduplication session")
        # controls_layout.addWidget(self.dedup_button)
        
        
        layout.addLayout(controls_layout)
        
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
        
        layout.addLayout(slider_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Create the three view tabs
        self.all_photos_view = AllPhotosView(self)
        self.grouped_photos_view = GroupedPhotosView(self)
        self.selected_photos_view = SelectedPhotosView(self)
        
        # Add tabs with icons and labels
        self.tab_widget.addTab(self.all_photos_view, "📷 All Photos")
        self.tab_widget.addTab(self.grouped_photos_view, "📁 Grouped")
        self.tab_widget.addTab(self.selected_photos_view, "⭐ Selected")
        
        # Connect selection signals for synchronization
        self.all_photos_view.selection_changed.connect(self.on_selection_changed_from_tab)
        self.selected_photos_view.selection_changed.connect(self.on_selection_changed_from_tab)
        self.grouped_photos_view.selection_changed.connect(self.on_selection_changed_from_tab)
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        layout.addWidget(self.tab_widget)
        
        # Keep reference to preview panel for compatibility
        self.preview_panel = self.grouped_photos_view.get_preview_panel()
        
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
            
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.folder_button.setEnabled(False)
        
        # Start processing thread
        threshold = self.threshold_slider.value() / 100.0
        self.processing_thread = ProcessingThread(self.current_folder, threshold)
        self.processing_thread.progress_updated.connect(self.on_progress_updated)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.start()
    
    def on_progress_updated(self, percentage: int, message: str):
        """Handle progress updates from processing thread."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
    
    def on_processing_finished(self, groups: List[List[str]], embeddings: Dict[str, np.ndarray]):
        """Handle completion of processing."""
        self.current_groups = groups
        self.current_embeddings = embeddings
        self.progress_bar.setVisible(False)
        self.folder_button.setEnabled(True)
        
        # Extract all image paths from groups
        self.all_image_paths = []
        for group in groups:
            self.all_image_paths.extend(group)
        self.all_image_paths.sort()
        
        if groups:
            min_display_size = self.min_group_spinbox.value()
            
            # Update all views
            self.all_photos_view.set_images(self.all_image_paths)
            self.grouped_photos_view.set_groups(groups, min_display_size)
            self.selected_photos_view.update_selected_images()
            
            total_images = sum(len(g) for g in groups)
            
            # Calculate actual displayed group count (accounting for singles consolidation)
            displayed_count = self._calculate_displayed_group_count(groups, min_display_size)
            
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
                cache_info = f" | {cached_count} cached embeddings"
            except Exception:
                cache_info = ""
            
            dedup_info = f" | {len(dedup_groups)} groups ready for deduplication" if dedup_groups else ""
            self.status_label.setText(f"Found {displayed_count} groups with {total_images} total images{cache_info}{dedup_info}")
        else:
            self.all_photos_view.clear()
            self.grouped_photos_view.clear()
            self.selected_photos_view.update_selected_images()
            # self.dedup_button.setEnabled(False)
            self.status_label.setText("No similar images found or processing failed")
    
    def on_threshold_changed(self):
        """Handle threshold slider changes."""
        threshold = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        
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
            groups = grouper.group_by_threshold(self.current_embeddings, threshold, min_group_size=1)
            
            # Sort clusters by inter-cluster similarity (if enabled)
            if hasattr(self, 'sort_clusters_checkbox') and self.sort_clusters_checkbox.isChecked():
                # Use same threshold for cluster sorting as for grouping
                sorted_groups = grouper.sort_clusters_by_similarity(groups, self.current_embeddings, threshold)
            else:
                sorted_groups = groups
            
            # Debug output to console
            print(f"Threshold: {threshold}, Groups: {len(sorted_groups)}, Total images: {sum(len(g) for g in sorted_groups)}")
            print(f"Group sizes: {[len(g) for g in sorted_groups]}")
            
            self.current_groups = sorted_groups
            min_display_size = self.min_group_spinbox.value()
            
            # Update grouped photos view
            self.grouped_photos_view.set_groups(sorted_groups, min_display_size)
            
            # Update deduplication button availability
            dedup_groups = [g for g in sorted_groups if len(g) > 1]
            # self.dedup_button.setEnabled(len(dedup_groups) > 0)
            
            # Calculate actual displayed group count (accounting for singles consolidation)
            displayed_count = self._calculate_displayed_group_count(sorted_groups, min_display_size)
            dedup_info = f" | {len(dedup_groups)} groups ready for deduplication" if dedup_groups else ""
            self.status_label.setText(f"Regrouped at {threshold:.2f}: {displayed_count} groups with {sum(len(g) for g in sorted_groups)} total images{dedup_info}")
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
    

    
    def update_view_selected_button(self):
        """Update the selected photos tab and count."""
        count = len(self.global_selected_images)
        # Update tab title with count
        self.tab_widget.setTabText(2, f"⭐ Selected ({count})")
        # Update selected photos view
        self.selected_photos_view.update_selected_images()
    
    def get_global_selected_images(self):
        """Get the set of globally selected image paths."""
        return self.global_selected_images.copy()
    
    def clear_global_selection(self):
        """Clear all global selections."""
        self.global_selected_images.clear()
        self.update_view_selected_button()
    
    def add_to_global_selection(self, image_path: str):
        """Add an image to global selection."""
        self.global_selected_images.add(image_path)
        self.update_view_selected_button()
    
    def remove_from_global_selection(self, image_path: str):
        """Remove an image from global selection."""
        self.global_selected_images.discard(image_path)
        self.update_view_selected_button()
    
    def on_selection_changed_from_tab(self, image_path: str, is_selected: bool):
        """Handle selection changes from any tab view."""
        # Update global selection state
        if is_selected:
            self.add_to_global_selection(image_path)
        else:
            self.remove_from_global_selection(image_path)
        
        # Sync selection across all tabs
        if self.tab_widget.currentWidget() != self.all_photos_view:
            self.all_photos_view.sync_selection(image_path, is_selected)
        if self.tab_widget.currentWidget() != self.selected_photos_view:
            self.selected_photos_view.sync_selection(image_path, is_selected)
        if self.tab_widget.currentWidget() != self.grouped_photos_view:
            self.grouped_photos_view.sync_selection(image_path, is_selected)
    
    def on_tab_changed(self, index: int):
        """Handle tab changes."""
        # Update selected photos view when switching to it
        if index == 2:  # Selected photos tab
            self.selected_photos_view.update_selected_images()
        # Update tab count
        count = len(self.global_selected_images)
        self.tab_widget.setTabText(2, f"⭐ Selected ({count})")
    
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