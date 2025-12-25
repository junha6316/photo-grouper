from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QLabel, QProgressBar,
    QMessageBox, QSplitter, QStackedWidget, QFrame, QComboBox, QToolButton, QMenu,
    QDialog, QDialogButtonBox, QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction
import os
from typing import List, Dict
import numpy as np

from core.scanner import ImageScanner

from core.grouper import PhotoGrouper
from ui.views import AllPhotosView, GroupedPhotosView, GroupDetailView, SelectedTray

DEFAULT_FOCUS_THRESHOLD = 100
DEFAULT_EXTREME_SIMILARITY_THRESHOLD = 0.98


class ElidedLabel(QLabel):
    """Single-line label that elides long text to avoid layout overflow."""

    def __init__(self, text: str = "", parent=None, elide_mode: Qt.TextElideMode = Qt.ElideMiddle):
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = elide_mode
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(0)
        if text:
            self.set_full_text(text)

    def set_full_text(self, text: str):
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._update_elided_text()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self):
        if not self._full_text:
            self.setText("")
            return
        metrics = self.fontMetrics()
        available = max(0, self.width() - 4)
        self.setText(metrics.elidedText(self._full_text, self._elide_mode, available))


class StartupDialog(QDialog):
    """Startup dialog for initial folder selection and focus filtering."""

    def __init__(self, parent=None, hide_out_of_focus: bool = True):
        super().__init__(parent)
        self.selected_folder = ""
        self.hide_out_of_focus = hide_out_of_focus
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Start Photo Grouper")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Choose a folder to start")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        folder_layout = QVBoxLayout()
        folder_layout.setSpacing(6)
        folder_row = QHBoxLayout()
        self.select_button = QPushButton("Select Folder")
        self.select_button.clicked.connect(self.select_folder)
        folder_row.addWidget(self.select_button)
        folder_row.addStretch()
        folder_layout.addLayout(folder_row)

        self.folder_label = ElidedLabel("No folder selected")
        self.folder_label.setStyleSheet("color: #666;")
        self.folder_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.folder_label.setMinimumWidth(0)
        folder_layout.addWidget(self.folder_label)
        layout.addLayout(folder_layout)

        self.hide_checkbox = QCheckBox("Hide out-of-focus photos")
        self.hide_checkbox.setChecked(self.hide_out_of_focus)
        self.hide_checkbox.stateChanged.connect(self.on_hide_changed)
        layout.addWidget(self.hide_checkbox)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        start_button = self.button_box.button(QDialogButtonBox.Ok)
        start_button.setText("Start")
        start_button.setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if folder:
            self.selected_folder = folder
            self.folder_label.set_full_text(folder)
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def on_hide_changed(self, state: int):
        self.hide_out_of_focus = state == Qt.Checked

class ProcessingThread(QThread):
    """Thread for background image processing."""
    
    progress_updated = Signal(int, str)  # progress percentage, status message
    processing_finished = Signal(list, dict, list, dict, list)  # groups, embeddings, similarities, focus_scores, filtered_out
    images_scanned = Signal(list, int)  # Emit scanned images immediately for display (paths, filtered_count)
    
    def __init__(self, folder_path: str, threshold: float, hide_out_of_focus: bool, focus_threshold: float):
        super().__init__()
        self.folder_path = folder_path
        self.threshold = threshold
        self.hide_out_of_focus = hide_out_of_focus
        self.focus_threshold = focus_threshold
        
    def run(self):
        try:
            # Step 1: Scan images
            self.progress_updated.emit(10, "Scanning for images...")
            scanner = ImageScanner()
            image_paths = scanner.scan_images(self.folder_path)
            image_paths.sort()
            if not image_paths:
                self.progress_updated.emit(100, "No images found")
                self.processing_finished.emit([], {}, [], {}, [])
                return

            focus_scores = {}
            filtered_out = []

            if self.hide_out_of_focus:
                from core.focus_detector import compute_focus_scores

                self.progress_updated.emit(12, f"Found {len(image_paths)} images, checking focus...")
                focus_start = 12
                focus_end = 18

                def focus_progress_callback(current_idx: int, total: int, current_path: str):
                    progress = focus_start + int((focus_end - focus_start) * current_idx / total)
                    self.progress_updated.emit(progress, f"Checking focus {current_idx}/{total}")

                focus_scores = compute_focus_scores(
                    image_paths,
                    progress_callback=focus_progress_callback,
                )
                threshold = float(self.focus_threshold)
                filtered_paths = []
                for path in image_paths:
                    score = focus_scores.get(path, 0.0)
                    if score >= threshold:
                        filtered_paths.append(path)
                    else:
                        filtered_out.append(path)

                if filtered_out:
                    self.progress_updated.emit(
                        focus_end,
                        f"Filtered {len(filtered_out)} out-of-focus images",
                    )

                image_paths = filtered_paths
                if not image_paths:
                    self.progress_updated.emit(100, "No in-focus images found")
                    self.images_scanned.emit([], len(filtered_out))
                    self.processing_finished.emit([], {}, [], focus_scores, filtered_out)
                    return
            else:
                self.progress_updated.emit(12, f"Found {len(image_paths)} images, extracting features...")

            # Emit scanned images immediately so UI can start displaying them
            self.images_scanned.emit(image_paths, len(filtered_out))
            
            # Step 2: Initialize embedder and fit PCA
            self.progress_updated.emit(19, "Initializing embedder...")
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
            self.processing_finished.emit(
                sorted_groups,
                embeddings,
                sorted_similarities,
                focus_scores,
                filtered_out,
            )
            
        except Exception as e:
            print(f"DEBUG: Error: {str(e)}")
            self.progress_updated.emit(100, f"Error: {str(e)}")
            self.processing_finished.emit([], {}, [], {}, [])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Grouper")
        self.setGeometry(100, 100, 1200, 800)
        self._startup_prompt_shown = False
        
        # Data
        self.current_folder = ""
        self.current_groups = []
        self.current_embeddings = {}
        self.all_image_paths = []  # Store all scanned images
        self.focus_scores = {}
        self.out_of_focus_images = []
        self.exclude_extremely_similar = False
        self.extremely_similar_threshold = DEFAULT_EXTREME_SIMILARITY_THRESHOLD
        
        # Global selection state - tracks selected images across all clusters
        self.global_selected_images = set()
        
        # Processing thread
        self.processing_thread = None

        # Splitter sizing state
        self._splitter_initialized = False
        self._splitter_adjusting = False
        self._selected_tray_visible = False
        self._left_panel_ratio = 0.24
        self._right_panel_ratio = 0.16
        self._left_panel_min = 240
        self._left_panel_max = 360
        self._right_panel_min = 200
        self._right_panel_max = 240
        
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
        from PySide6.QtWidgets import QSpinBox
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

        # Focus filtering controls
        focus_controls_layout = QHBoxLayout()
        self.hide_out_of_focus_checkbox = QCheckBox("Hide out-of-focus photos")
        self.hide_out_of_focus_checkbox.setChecked(True)
        self.hide_out_of_focus_checkbox.setToolTip("Hide blurry images based on sharpness score")
        self.hide_out_of_focus_checkbox.stateChanged.connect(self.on_focus_filter_changed)
        focus_controls_layout.addWidget(self.hide_out_of_focus_checkbox)

        focus_controls_layout.addStretch()

        focus_controls_layout.addWidget(QLabel("Focus threshold:"))
        self.focus_threshold_slider = QSlider(Qt.Horizontal)
        self.focus_threshold_slider.setMinimum(10)
        self.focus_threshold_slider.setMaximum(500)
        self.focus_threshold_slider.setValue(DEFAULT_FOCUS_THRESHOLD)
        self.focus_threshold_slider.valueChanged.connect(self.on_focus_threshold_changed)
        self.focus_threshold_slider.sliderReleased.connect(self.on_focus_threshold_released)
        focus_controls_layout.addWidget(self.focus_threshold_slider)

        self.focus_threshold_label = QLabel(str(DEFAULT_FOCUS_THRESHOLD))
        self.focus_threshold_label.setMinimumWidth(40)
        focus_controls_layout.addWidget(self.focus_threshold_label)

        advanced_layout.addLayout(focus_controls_layout)
        self._update_focus_controls_enabled()
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

        self.selection_mode_button = QPushButton("Selection Mode (Group)")
        self.selection_mode_button.setEnabled(False)
        self.selection_mode_button.clicked.connect(self.open_selection_mode)
        self.selection_mode_button.setToolTip("Select a group to enable selection mode")
        self.selection_mode_button.setStyleSheet("""
            QPushButton {
                padding: 4px 10px;
                font-size: 11px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f8f8;
            }
            QPushButton:hover:enabled {
                background-color: #e8e8e8;
            }
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
            }
        """)
        view_layout.addWidget(self.selection_mode_button)

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
        self.grouped_photos_view.exclude_similar_toggled.connect(self.on_exclude_similar_toggled)

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
        self.content_splitter.splitterMoved.connect(self._on_splitter_moved)

        self.grouped_photos_view.setMinimumWidth(self._left_panel_min)
        self.selected_tray.setMinimumWidth(self._right_panel_min)
        # Initial layout: 25% left panel, 75% center (selected tray hidden initially)
        self._apply_splitter_sizes()

        layout.addWidget(self.content_splitter, 1)

        # Keep reference to preview panel for compatibility
        self.preview_panel = self.grouped_photos_view.get_preview_panel()

        self.view_mode = ""
        self.set_view_mode("grouped")

    def showEvent(self, event):
        super().showEvent(event)
        if not self._splitter_initialized:
            self._splitter_initialized = True
            self._apply_splitter_sizes()
        if not self._startup_prompt_shown:
            self._startup_prompt_shown = True
            QTimer.singleShot(0, self._show_startup_dialog)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._splitter_initialized:
            self._apply_splitter_sizes()

    def _show_startup_dialog(self):
        dialog = StartupDialog(self, hide_out_of_focus=self.hide_out_of_focus_checkbox.isChecked())
        if dialog.exec() == QDialog.Accepted and dialog.selected_folder:
            self.hide_out_of_focus_checkbox.setChecked(dialog.hide_out_of_focus)
            self.current_folder = dialog.selected_folder
            self.folder_label.setText(f"Folder: {os.path.basename(self.current_folder)}")
            self.process_folder()
        
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
        self._update_selected_tray_visibility()
        self._update_selection_mode_enabled()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.folder_button.setEnabled(False)
        
        # Start processing thread
        threshold = self.threshold_slider.value() / 100.0
        hide_out_of_focus = self.hide_out_of_focus_checkbox.isChecked()
        focus_threshold = float(self.focus_threshold_slider.value())
        self.processing_thread = ProcessingThread(
            self.current_folder,
            threshold,
            hide_out_of_focus,
            focus_threshold,
        )
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
    
    def on_images_scanned(self, image_paths: List[str], hidden_count: int):
        """Handle immediate display of scanned images before processing."""
        # Store the scanned images
        self.all_image_paths = image_paths
        
        # Immediately populate the All Photos view
        self.all_photos_view.set_images(image_paths, hidden_count)
        self.set_view_mode("all")
        self._update_selection_mode_enabled()

        # Keep grouped view showing processing with initial status
        focus_note = ""
        if hidden_count:
            focus_note = f" ({hidden_count} hidden out-of-focus)"
        self.grouped_photos_view.show_processing_message(
            f"Found {len(image_paths)} images{focus_note}\nExtracting image features..."
        )
    
    def on_processing_finished(
        self,
        groups: List[List[str]],
        embeddings: Dict[str, np.ndarray],
        similarities: List[float],
        focus_scores: Dict[str, float],
        filtered_out: List[str],
    ):
        """Handle completion of processing."""
        self.current_groups = groups
        self.current_embeddings = embeddings
        self.current_similarities = similarities
        self.focus_scores = focus_scores or {}
        self.out_of_focus_images = filtered_out or []
        self.progress_bar.setVisible(False)
        self.folder_button.setEnabled(True)

        hidden_count = len(self.out_of_focus_images)
        if hidden_count:
            for image_path in list(self.global_selected_images):
                if image_path in self.out_of_focus_images:
                    self.remove_from_global_selection(image_path)
        
        # Extract all image paths from groups
        self.all_image_paths = []
        for group in groups:
            self.all_image_paths.extend(group)
        self.all_image_paths.sort()
        
        if groups:
            # Update all views (All photos already set in on_images_scanned)
            # Just update the grouped view with the final groups
            self.selected_tray.sync_selected_images(self.global_selected_images)
            self._update_selected_tray_visibility()
            
            self._refresh_grouped_display()
            self._update_selection_mode_enabled()
            
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
            if hidden_count:
                status_parts.append(f"{hidden_count} hidden out-of-focus")
            self.status_label.setText(" | ".join(status_parts))
        else:
            self.all_photos_view.clear()
            self.grouped_photos_view.clear()
            self.group_detail_view.clear()
            self.selected_tray.sync_selected_images(self.global_selected_images)
            self._update_selected_tray_visibility()
            # self.dedup_button.setEnabled(False)
            if hidden_count and not embeddings:
                self.grouped_photos_view.set_summary_text("No in-focus images found")
                self.status_label.setText(f"Filtered {hidden_count} out-of-focus images")
            else:
                self.grouped_photos_view.set_summary_text("No similar images found")
                status_text = "No similar images found or processing failed"
                if hidden_count:
                    status_text = f"{status_text} | {hidden_count} hidden out-of-focus"
                self.status_label.setText(status_text)
            self.grouped_photos_view.sync_selected_counts(self.global_selected_images)
            self._update_selection_mode_enabled()
    
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

    def _update_focus_controls_enabled(self):
        enabled = self.hide_out_of_focus_checkbox.isChecked()
        self.focus_threshold_slider.setEnabled(enabled)
        self.focus_threshold_label.setEnabled(enabled)

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

    def on_focus_filter_changed(self):
        """Handle focus filter toggle."""
        self._update_focus_controls_enabled()
        if self.processing_thread and self.processing_thread.isRunning():
            return
        if self.current_folder:
            self.process_folder()

    def on_focus_threshold_changed(self):
        """Handle focus threshold slider changes."""
        value = self.focus_threshold_slider.value()
        self.focus_threshold_label.setText(str(value))

    def on_focus_threshold_released(self):
        """Apply focus threshold changes after slider release."""
        if not self.hide_out_of_focus_checkbox.isChecked():
            return
        if self.processing_thread and self.processing_thread.isRunning():
            return
        if self.current_folder:
            self.process_folder()
    
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
            
            # Update grouped photos view
            self._refresh_grouped_display()
            
            # Update deduplication button availability
            dedup_groups = [g for g in sorted_groups if len(g) > 1]
            # self.dedup_button.setEnabled(len(dedup_groups) > 0)
            
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

    def _on_splitter_moved(self, pos: int, index: int):
        if self._splitter_adjusting:
            return
        sizes = self.content_splitter.sizes()
        total_width = sum(sizes)
        if total_width <= 0:
            return
        self._left_panel_ratio = sizes[0] / total_width
        if sizes[2] > 0:
            self._right_panel_ratio = sizes[2] / total_width

    def _get_splitter_total_width(self) -> int:
        sizes = self.content_splitter.sizes()
        total_width = sum(sizes)
        if total_width <= 0:
            total_width = self.content_splitter.width()
        if total_width <= 0:
            total_width = self.width()
        return total_width

    def _apply_splitter_sizes(self):
        if not hasattr(self, "content_splitter"):
            return
        total_width = self._get_splitter_total_width()
        if total_width <= 0:
            return

        has_tray = self._selected_tray_visible
        left_width = int(total_width * self._left_panel_ratio)
        left_width = max(self._left_panel_min, min(self._left_panel_max, left_width))

        right_width = 0
        if has_tray:
            right_width = int(total_width * self._right_panel_ratio)
            right_width = max(self._right_panel_min, min(self._right_panel_max, right_width))

        if left_width + right_width > total_width:
            left_width = max(self._left_panel_min, total_width - right_width)
            if left_width + right_width > total_width:
                right_width = max(0, total_width - left_width)

        center_width = max(0, total_width - left_width - right_width)

        self._splitter_adjusting = True
        try:
            self.content_splitter.setSizes([left_width, center_width, right_width])
        finally:
            self._splitter_adjusting = False

    def _update_selected_tray_visibility(self):
        """Show or hide the selected tray based on whether images are selected."""
        has_selections = len(self.global_selected_images) > 0
        if has_selections == self._selected_tray_visible:
            self._apply_splitter_sizes()
            return

        self._selected_tray_visible = has_selections
        self._apply_splitter_sizes()

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

    def on_exclude_similar_toggled(self, enabled: bool):
        """Handle excluding extremely similar images from the group list."""
        self.exclude_extremely_similar = enabled
        self._refresh_grouped_display(preserve_detail=True)

    def _filter_group_extremely_similar(self, group: List[str]) -> List[str]:
        if len(group) <= 1 or not self.current_embeddings:
            return group
        kept = []
        for image_path in group:
            embedding = self.current_embeddings.get(image_path)
            if embedding is None:
                kept.append(image_path)
                continue
            is_duplicate = False
            for kept_path in kept:
                kept_embedding = self.current_embeddings.get(kept_path)
                if kept_embedding is None:
                    continue
                similarity = float(np.dot(embedding, kept_embedding))
                if similarity >= self.extremely_similar_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(image_path)
        if not kept and group:
            kept = [group[0]]
        return kept

    def _get_display_groups(
        self,
        groups: List[List[str]],
        similarities: List[float],
    ) -> tuple[list, list]:
        if not self.exclude_extremely_similar:
            return groups, similarities
        filtered_groups = [
            self._filter_group_extremely_similar(group) for group in groups
        ]
        return filtered_groups, similarities

    def _refresh_grouped_display(self, preserve_detail: bool = False):
        if not self.current_groups:
            self.grouped_photos_view.clear()
            if not preserve_detail:
                self.group_detail_view.clear()
            return
        min_display_size = self.min_group_spinbox.value()

        ordered_entries = []
        for idx, group in enumerate(self.current_groups):
            similarity = None
            if self.current_similarities and idx < len(self.current_similarities):
                similarity = self.current_similarities[idx]
            is_singles = similarity == 0.0 if similarity is not None else False
            ordered_entries.append(
                {
                    "group": group,
                    "similarity": similarity,
                    "is_singles": is_singles,
                }
            )

        non_singles = [entry for entry in ordered_entries if not entry["is_singles"]]
        singles = [entry for entry in ordered_entries if entry["is_singles"]]
        non_singles.sort(key=lambda entry: len(entry["group"]), reverse=True)
        ordered_entries = non_singles + singles

        visible_entries = [
            entry for entry in ordered_entries
            if len(entry["group"]) >= min_display_size
        ]

        display_groups = []
        display_similarities = []
        for entry in visible_entries:
            group = entry["group"]
            if self.exclude_extremely_similar:
                group = self._filter_group_extremely_similar(group)
            display_groups.append(group)
            display_similarities.append(entry["similarity"])

        self.grouped_photos_view.set_groups(
            display_groups,
            1,
            display_similarities,
            preserve_order=True,
        )
        if not preserve_detail:
            self.group_detail_view.clear()

        total_images = sum(len(group) for group in display_groups)
        displayed_count = len(visible_entries)
        summary_text = f"Found {displayed_count} groups with {total_images} total images"
        hidden_count = len(self.out_of_focus_images)
        if hidden_count:
            summary_text += f" | Hidden {hidden_count} out-of-focus"
        if self.exclude_extremely_similar:
            summary_text += (
                f" | Excluding >= {self.extremely_similar_threshold:.2f} similar"
            )
        self.grouped_photos_view.set_summary_text(summary_text)
        self.grouped_photos_view.sync_selected_counts(self.global_selected_images)
        if preserve_detail:
            self._refresh_open_group_detail()
        self._update_selection_mode_enabled()

    def _refresh_open_group_detail(self):
        if not self.group_detail_view.cluster_images:
            return
        seed_image = self.group_detail_view.cluster_images[0]
        source_group = None
        source_index = None
        for idx, group in enumerate(self.current_groups):
            if seed_image in group:
                source_group = group
                source_index = idx
                break
        if not source_group:
            return
        similarity = None
        if self.current_similarities and source_index is not None:
            if source_index < len(self.current_similarities):
                similarity = self.current_similarities[source_index]
        is_singles = similarity == 0.0 if similarity is not None else False
        display_group = source_group
        if self.exclude_extremely_similar:
            display_group = self._filter_group_extremely_similar(source_group)
        self.group_detail_view.update_cluster_images(
            display_group,
            similarity=similarity,
            is_singles_group=is_singles,
        )

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
        self._update_selection_mode_enabled()

    def open_selection_mode(self):
        """Open swipe-based selection mode."""
        cluster_images = []
        if hasattr(self.group_detail_view, "cluster_images"):
            cluster_images = list(self.group_detail_view.cluster_images)
        if not cluster_images:
            QMessageBox.information(
                self,
                "Select a Group",
                "Select a group in the list before starting selection mode.",
            )
            return
        from core.review_session import ReviewSession
        from ui.swipe_review_dialog import SwipeReviewDialog
        session = ReviewSession(self.current_folder, cluster_images)
        title = "Selection Mode"
        if getattr(self.group_detail_view, "is_singles_group", False):
            title = "Selection Mode - Singles"
        elif getattr(self.group_detail_view, "cluster_number", 0):
            title = f"Selection Mode - Group {self.group_detail_view.cluster_number}"
        dialog = SwipeReviewDialog(session, self, title=title)
        dialog.exec()

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

    def _update_selection_mode_enabled(self):
        if hasattr(self, "selection_mode_button"):
            has_group = bool(getattr(self.group_detail_view, "cluster_images", []))
            self.selection_mode_button.setEnabled(has_group)
            if has_group:
                self.selection_mode_button.setToolTip("Open selection mode for this group")
            else:
                self.selection_mode_button.setToolTip("Select a group to enable selection mode")
    
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
        self._update_selected_tray_visibility()
    
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
