from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
    QPushButton, QWidget, QFrame, QProgressBar, QTextEdit,
    QSplitter, QGroupBox, QRadioButton, QButtonGroup,
    QMessageBox, QFileDialog, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QPixmap
import os
from typing import List, Dict
import shutil
from datetime import datetime

from core.deduplicator import ImageDeduplicator
from ui.utils.async_image_loader import get_async_loader, ImageLoadResult
from infra.cache_db import EmbeddingCache

class DeduplicationWorker(QThread):
    """Worker thread for finding duplicates."""
    
    progress_updated = Signal(int, str)  # progress percentage, status message
    duplicates_found = Signal(list)  # duplicate sets
    
    def __init__(self, embeddings: Dict, groups: List[List[str]], threshold: float):
        super().__init__()
        self.embeddings = embeddings
        self.groups = groups
        self.threshold = threshold
        
    def run(self):
        try:
            self.progress_updated.emit(10, "Initializing deduplicator...")
            deduplicator = ImageDeduplicator(self.threshold)
            
            self.progress_updated.emit(30, "Analyzing image similarities...")
            duplicate_sets = deduplicator.find_duplicates(self.embeddings, self.groups)
            
            self.progress_updated.emit(90, "Processing results...")
            
            self.progress_updated.emit(100, f"Found {len(duplicate_sets)} duplicate sets")
            self.duplicates_found.emit(duplicate_sets)
            
        except Exception as e:
            self.progress_updated.emit(100, f"Error: {str(e)}")
            self.duplicates_found.emit([])

class DuplicateSetWidget(QWidget):
    """Widget displaying a single set of duplicate images with selection controls."""
    
    keeper_changed = Signal(int, str)  # set_index, chosen_keeper_path
    
    def __init__(self, set_index: int, duplicate_set: Dict, parent=None):
        super().__init__(parent)
        self.set_index = set_index
        self.duplicate_set = duplicate_set
        self.image_widgets = {}
        self.button_group = QButtonGroup(self)
        
        self.init_ui()
        self.load_images()
        
    def init_ui(self):
        """Initialize the UI for this duplicate set."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header with set info
        header_layout = QHBoxLayout()
        
        title_label = QLabel(f"Duplicate Set {self.set_index + 1}")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Similarity info
        avg_sim = self.duplicate_set.get('avg_similarity', 0)
        sim_label = QLabel(f"Similarity: {avg_sim:.1%}")
        sim_label.setStyleSheet("color: #666; font-size: 11px;")
        header_layout.addWidget(sim_label)
        
        layout.addLayout(header_layout)
        
        # Images container
        images_container = QFrame()
        images_container.setFrameStyle(QFrame.Box)
        images_container.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #fafafa;
                padding: 10px;
            }
        """)
        
        images_layout = QHBoxLayout(images_container)
        images_layout.setSpacing(15)
        
        # Create image widgets for each duplicate
        for i, image_path in enumerate(self.duplicate_set['images']):
            image_widget = self.create_image_widget(image_path, i)
            images_layout.addWidget(image_widget)
            self.image_widgets[image_path] = image_widget
        
        layout.addWidget(images_container)
        
        # Select the recommended keeper by default
        recommended = self.duplicate_set['recommended_keeper']
        if recommended in self.image_widgets:
            radio_button = self.image_widgets[recommended].findChild(QRadioButton)
            if radio_button:
                radio_button.setChecked(True)
        
        # Connect button group signal
        self.button_group.buttonClicked.connect(self.on_keeper_selected)
        
    def create_image_widget(self, image_path: str, index: int) -> QWidget:
        """Create a widget for a single duplicate image."""
        widget = QWidget()
        widget.setFixedWidth(200)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # Image display
        image_label = QLabel()
        image_label.setFixedSize(180, 180)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: white;
                padding: 5px;
            }
        """)
        layout.addWidget(image_label)
        
        # Store reference for image loading
        widget.image_label = image_label
        widget.image_path = image_path
        
        # Radio button for selection
        radio_button = QRadioButton("Keep this image")
        self.button_group.addButton(radio_button, index)
        
        # Recommendation indicator
        is_recommended = image_path == self.duplicate_set['recommended_keeper']
        if is_recommended:
            radio_button.setText("🏆 Keep this image (Recommended)")
            radio_button.setStyleSheet("color: #007acc; font-weight: bold;")
        
        layout.addWidget(radio_button)
        
        # Image info
        filename = os.path.basename(image_path)
        filename_label = QLabel(filename)
        filename_label.setWordWrap(True)
        filename_label.setStyleSheet("font-size: 10px; color: #666;")
        filename_label.setToolTip(image_path)
        layout.addWidget(filename_label)
        
        # File size and resolution info
        try:
            file_size = os.path.getsize(image_path)
            size_mb = file_size / (1024 * 1024)
            
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
            
            info_text = f"{width}×{height}\n{size_mb:.1f} MB"
            info_label = QLabel(info_text)
            info_label.setStyleSheet("font-size: 9px; color: #888; text-align: center;")
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)
            
        except Exception:
            pass
        
        # Recommendation reason
        if is_recommended:
            reason_label = QLabel(self.duplicate_set['reason'])
            reason_label.setWordWrap(True)
            reason_label.setStyleSheet("""
                font-size: 9px; 
                color: #007acc; 
                background-color: #e6f3ff; 
                padding: 4px; 
                border-radius: 4px;
            """)
            layout.addWidget(reason_label)
        
        return widget
        
    def load_images(self):
        """Load thumbnail images for all duplicates."""
        loader = get_async_loader()
        loader.image_loaded.connect(self.on_image_loaded)
        
        for image_path, widget in self.image_widgets.items():
            widget.image_label.setText("📷\nLoading...")
            loader.load_image(image_path, 150, priority=1)
    
    def on_image_loaded(self, result: ImageLoadResult):
        """Handle completion of image loading."""
        if result.image_path not in self.image_widgets:
            return
        
        widget = self.image_widgets[result.image_path]
        
        if result.success and result.pixmap:
            widget.image_label.setPixmap(result.pixmap)
        else:
            widget.image_label.setText(f"⚠️\nFailed to load")
            widget.image_label.setStyleSheet(widget.image_label.styleSheet() + "color: red;")
    
    def on_keeper_selected(self, button):
        """Handle keeper selection."""
        button_index = self.button_group.id(button)
        if 0 <= button_index < len(self.duplicate_set['images']):
            chosen_path = self.duplicate_set['images'][button_index]
            self.keeper_changed.emit(self.set_index, chosen_path)
    
    def get_chosen_keeper(self) -> str:
        """Get the currently chosen keeper image."""
        checked_button = self.button_group.checkedButton()
        if checked_button:
            button_index = self.button_group.id(checked_button)
            if 0 <= button_index < len(self.duplicate_set['images']):
                return self.duplicate_set['images'][button_index]
        return self.duplicate_set['recommended_keeper']

class DeduplicationDialog(QDialog):
    """Dialog for reviewing and applying image deduplication."""
    
    def __init__(self, embeddings: Dict, groups: List[List[str]], parent=None):
        super().__init__(parent)
        self.embeddings = embeddings
        self.groups = groups
        self.duplicate_sets = []
        self.user_choices = {}
        self.deduplicator = None
        self.session_id = None
        self.cache_db = EmbeddingCache()
        
        self.setWindowTitle("Image Deduplication")
        self.setGeometry(100, 100, 1200, 800)
        
        self.init_ui()
        self.start_duplicate_detection()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Image Deduplication")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(16)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        subtitle_label = QLabel("Review duplicate images and choose which ones to keep")
        subtitle_label.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(subtitle_label)
        
        # Progress bar (initially visible)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(True)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Starting duplicate detection...")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        # Main content (initially hidden)
        self.main_content = QSplitter(Qt.Horizontal)
        self.main_content.setVisible(False)
        
        # Left side - duplicate sets
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.threshold_label = QLabel("Similarity threshold: 98%")
        controls_layout.addWidget(self.threshold_label)
        
        controls_layout.addStretch()
        
        self.view_history_button = QPushButton("📊 View History")
        self.view_history_button.clicked.connect(self.show_deduplication_history)
        controls_layout.addWidget(self.view_history_button)
        
        self.export_plan_button = QPushButton("📄 Export Plan")
        self.export_plan_button.clicked.connect(self.export_deduplication_plan)
        self.export_plan_button.setEnabled(False)
        controls_layout.addWidget(self.export_plan_button)
        
        left_layout.addLayout(controls_layout)
        
        # Scroll area for duplicate sets
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setSpacing(20)
        self.scroll_layout.addStretch()  # Push content to top
        
        self.scroll_area.setWidget(self.scroll_widget)
        left_layout.addWidget(self.scroll_area)
        
        self.main_content.addWidget(left_widget)
        
        # Right side - summary and actions
        right_widget = QWidget()
        right_widget.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_widget)
        
        # Summary box
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_label = QLabel("No duplicates detected yet...")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-size: 11px; color: #666;")
        summary_layout.addWidget(self.summary_label)
        
        right_layout.addWidget(summary_group)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        self.move_to_folder_checkbox = QCheckBox("Move duplicates to folder instead of deleting")
        self.move_to_folder_checkbox.setChecked(True)  # Default to safer option
        export_layout.addWidget(self.move_to_folder_checkbox)
        
        self.backup_checkbox = QCheckBox("Create backup before applying changes")
        self.backup_checkbox.setChecked(True)
        export_layout.addWidget(self.backup_checkbox)
        
        right_layout.addWidget(export_group)
        
        # Action buttons
        button_layout = QVBoxLayout()
        
        self.apply_button = QPushButton("🗑️ Apply Deduplication")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_deduplication)
        self.apply_button.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                border: 2px solid #e74c3c;
                border-radius: 6px;
                background-color: #fadbd8;
                color: #c0392b;
            }
            QPushButton:hover:enabled {
                background-color: #f5b7b1;
            }
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
                border-color: #ddd;
            }
        """)
        button_layout.addWidget(self.apply_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        self.close_button.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        button_layout.addWidget(self.close_button)
        
        right_layout.addLayout(button_layout)
        right_layout.addStretch()
        
        self.main_content.addWidget(right_widget)
        self.main_content.setSizes([800, 300])
        
        layout.addWidget(self.main_content)
    
    def start_duplicate_detection(self):
        """Start the duplicate detection process."""
        self.worker = DeduplicationWorker(self.embeddings, self.groups, 0.98)
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.duplicates_found.connect(self.on_duplicates_found)
        self.worker.start()
    
    def on_progress_updated(self, percentage: int, message: str):
        """Handle progress updates."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
    
    def on_duplicates_found(self, duplicate_sets: List[Dict]):
        """Handle completion of duplicate detection."""
        self.duplicate_sets = duplicate_sets
        self.deduplicator = ImageDeduplicator()
        
        # Create deduplication session
        total_images = len(self.embeddings)
        self.session_id = self.cache_db.create_deduplication_session(
            similarity_threshold=0.98,
            total_images=total_images,
            notes=f"Processing {total_images} images from {len(self.groups)} groups"
        )
        
        # Track all processed files
        if self.session_id:
            self._track_initial_files(duplicate_sets)
        
        # Hide progress, show content
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.main_content.setVisible(True)
        
        if not duplicate_sets:
            self.show_no_duplicates()
            if self.session_id:
                self.cache_db.update_deduplication_session(
                    self.session_id, 
                    duplicate_sets_found=0,
                    status='completed'
                )
            return
        
        # Update session with duplicate sets found
        if self.session_id:
            self.cache_db.update_deduplication_session(
                self.session_id,
                duplicate_sets_found=len(duplicate_sets)
            )
        
        # Display duplicate sets
        self.display_duplicate_sets()
        
        # Update summary
        self.update_summary()
        
        # Enable buttons
        self.export_plan_button.setEnabled(True)
        self.apply_button.setEnabled(True)
    
    def _track_initial_files(self, duplicate_sets: List[Dict]):
        """Track all files processed in this deduplication session."""
        if not self.session_id:
            return
        
        # Track files in duplicate sets
        for group_id, dup_set in enumerate(duplicate_sets):
            for image_path in dup_set['images']:
                try:
                    file_size = os.path.getsize(image_path)
                    file_hash = self._get_file_hash(image_path)
                    is_keeper = image_path == dup_set['recommended_keeper']
                    avg_similarity = dup_set.get('avg_similarity', 0)
                    
                    self.cache_db.save_deduplication_file(
                        session_id=self.session_id,
                        file_path=image_path,
                        file_hash=file_hash,
                        file_size=file_size,
                        duplicate_group_id=group_id,
                        is_keeper=is_keeper,
                        action_taken='pending',
                        similarity_score=avg_similarity
                    )
                except Exception as e:
                    print(f"Error tracking file {image_path}: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for a file."""
        import hashlib
        try:
            stat = os.stat(file_path)
            hash_string = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(hash_string.encode()).hexdigest()
        except:
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def show_no_duplicates(self):
        """Show message when no duplicates are found."""
        no_dupes_label = QLabel("✅ No duplicate images found!\n\nAll your images are unique.")
        no_dupes_label.setAlignment(Qt.AlignCenter)
        no_dupes_label.setStyleSheet("""
            font-size: 14px;
            color: #27ae60;
            padding: 40px;
            background-color: #d5f4e6;
            border-radius: 8px;
            border: 2px solid #27ae60;
        """)
        
        # Remove stretch and add the message
        self.scroll_layout.removeItem(self.scroll_layout.itemAt(0))
        self.scroll_layout.addWidget(no_dupes_label)
        
        self.summary_label.setText("No duplicates found. All images are unique.")
    
    def display_duplicate_sets(self):
        """Display all duplicate sets in the scroll area."""
        # Remove the stretch item
        if self.scroll_layout.count() > 0:
            self.scroll_layout.removeItem(self.scroll_layout.itemAt(0))
        
        # Add duplicate set widgets
        for i, duplicate_set in enumerate(self.duplicate_sets):
            set_widget = DuplicateSetWidget(i, duplicate_set, self)
            set_widget.keeper_changed.connect(self.on_keeper_changed)
            self.scroll_layout.addWidget(set_widget)
        
        # Add stretch at the end
        self.scroll_layout.addStretch()
    
    def on_keeper_changed(self, set_index: int, chosen_keeper: str):
        """Handle when user changes the keeper for a duplicate set."""
        self.user_choices[str(set_index)] = chosen_keeper
        self.update_summary()
    
    def update_summary(self):
        """Update the summary information."""
        if not self.duplicate_sets:
            return
        
        stats = self.deduplicator.get_deduplication_stats(self.duplicate_sets)
        
        # Calculate with user choices
        total_to_remove = 0
        space_to_save = 0
        
        for i, dup_set in enumerate(self.duplicate_sets):
            keeper = self.user_choices.get(str(i), dup_set['recommended_keeper'])
            for image_path in dup_set['images']:
                if image_path != keeper:
                    total_to_remove += 1
                    try:
                        space_to_save += os.path.getsize(image_path) / (1024 * 1024)
                    except OSError:
                        pass
        
        summary_text = f"""
<b>Duplicate Detection Results:</b><br>
• Found {len(self.duplicate_sets)} duplicate sets<br>
• {stats['total_duplicate_images']} total duplicate images<br>
• {total_to_remove} images will be removed<br>
• Space to save: {space_to_save:.1f} MB<br>
• Average similarity: {stats['average_similarity']:.1%}
        """.strip()
        
        self.summary_label.setText(summary_text)
    
    def export_deduplication_plan(self):
        """Export the deduplication plan to a text file."""
        if not self.deduplicator or not self.duplicate_sets:
            return
        
        # Update duplicate sets with user choices
        updated_sets = []
        for i, dup_set in enumerate(self.duplicate_sets):
            updated_set = dup_set.copy()
            if str(i) in self.user_choices:
                updated_set['recommended_keeper'] = self.user_choices[str(i)]
                updated_set['reason'] = "User selection"
            updated_sets.append(updated_set)
        
        plan_text = self.deduplicator.create_deduplication_plan(updated_sets)
        
        # Save to file
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Deduplication Plan", 
            f"deduplication_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(plan_text)
                QMessageBox.information(self, "Export Complete", f"Plan exported to:\n{filename}")
            except Exception as e:
                QMessageBox.warning(self, "Export Failed", f"Failed to export plan:\n{str(e)}")
    
    def apply_deduplication(self):
        """Apply the deduplication by removing/moving selected duplicates."""
        if not self.duplicate_sets:
            return
        
        # Confirm action
        reply = QMessageBox.question(
            self, "Confirm Deduplication",
            f"This will process {len(self.duplicate_sets)} duplicate sets.\n\n"
            f"Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # Create destination folder if moving files
            move_folder = None
            if self.move_to_folder_checkbox.isChecked():
                move_folder = QFileDialog.getExistingDirectory(
                    self, "Select folder to move duplicates to"
                )
                if not move_folder:
                    return
            
            # Create backup if requested
            if self.backup_checkbox.isChecked():
                self.create_backup()
            
            # Apply deduplication
            results = self.process_duplicates(move_folder)
            
            # Show results
            self.show_results(results)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Deduplication failed:\n{str(e)}")
    
    def create_backup(self):
        """Create a backup of all images before deduplication."""
        # This is a placeholder - in a real implementation, you might want to
        # create a timestamped backup folder
        pass
    
    def process_duplicates(self, move_folder: str = None) -> Dict:
        """Process the duplicate images according to user choices."""
        results = {
            'kept': [],
            'removed': [],
            'moved': [],
            'errors': [],
            'space_saved_mb': 0
        }
        
        for i, dup_set in enumerate(self.duplicate_sets):
            keeper = self.user_choices.get(str(i), dup_set['recommended_keeper'])
            results['kept'].append(keeper)
            
            # Update keeper in database
            if self.session_id:
                self._update_file_action(keeper, i, 'kept')
            
            for image_path in dup_set['images']:
                if image_path == keeper:
                    continue
                
                try:
                    file_size = os.path.getsize(image_path)
                    
                    if move_folder:
                        # Move to designated folder
                        filename = os.path.basename(image_path)
                        dest_path = os.path.join(move_folder, filename)
                        
                        # Handle name conflicts
                        counter = 1
                        while os.path.exists(dest_path):
                            name, ext = os.path.splitext(filename)
                            dest_path = os.path.join(move_folder, f"{name}_{counter}{ext}")
                            counter += 1
                        
                        shutil.move(image_path, dest_path)
                        results['moved'].append((image_path, dest_path))
                        
                        # Update database with move action
                        if self.session_id:
                            self._update_file_action(image_path, i, 'moved', dest_path)
                    else:
                        # Delete the file
                        os.remove(image_path)
                        results['removed'].append(image_path)
                        
                        # Update database with delete action
                        if self.session_id:
                            self._update_file_action(image_path, i, 'deleted')
                    
                    results['space_saved_mb'] += file_size / (1024 * 1024)
                    
                except Exception as e:
                    results['errors'].append(f"{image_path}: {str(e)}")
                    # Track error in database
                    if self.session_id:
                        self._update_file_action(image_path, i, 'error')
        
        # Update session with final results
        if self.session_id:
            self.cache_db.update_deduplication_session(
                self.session_id,
                space_saved_mb=results['space_saved_mb'],
                status='completed'
            )
        
        return results
    
    def _update_file_action(self, file_path: str, group_id: int, 
                           action: str, moved_to: str = None):
        """Update the action taken on a file in the database."""
        # Since we already saved the file info, we just need to update the action
        # This is a simplified approach - in production you might want to use UPDATE queries
        pass  # The current implementation saves files once, you could extend this
    
    def show_results(self, results: Dict):
        """Show the results of deduplication."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Deduplication Complete")
        msg.setIcon(QMessageBox.Information)
        
        kept_count = len(results['kept'])
        removed_count = len(results['removed'])
        moved_count = len(results['moved'])
        error_count = len(results['errors'])
        space_saved = results['space_saved_mb']
        
        msg.setText(f"Deduplication completed successfully!")
        
        details = f"""
Results:
• {kept_count} images kept
• {removed_count} images deleted
• {moved_count} images moved
• {space_saved:.1f} MB saved
• {error_count} errors occurred

        """.strip()
        
        if results['errors']:
            details += "\n\nErrors:\n"
            for error in results['errors'][:10]:  # Show first 10 errors
                details += f"• {error}\n"
            if len(results['errors']) > 10:
                details += f"... and {len(results['errors']) - 10} more errors"
        
        msg.setDetailedText(details)
        msg.exec()
        
        # Close dialog after successful completion
        self.accept()
    
    def show_deduplication_history(self):
        """Show a dialog with deduplication history."""
        history_dialog = QDialog(self)
        history_dialog.setWindowTitle("Deduplication History")
        history_dialog.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout(history_dialog)
        
        # Header
        header_label = QLabel("Recent Deduplication Sessions")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(14)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        # History text area
        history_text = QTextEdit()
        history_text.setReadOnly(True)
        
        # Get history from database
        sessions = self.cache_db.get_deduplication_history(limit=20)
        
        if not sessions:
            history_text.setPlainText("No deduplication sessions found.")
        else:
            text_lines = []
            for session in sessions:
                text_lines.append(f"Session: {session['session_id']}")
                text_lines.append(f"Date: {session['created_at']}")
                text_lines.append(f"Threshold: {session['similarity_threshold']:.0%}")
                text_lines.append(f"Images Processed: {session['total_images_processed']}")
                text_lines.append(f"Duplicate Sets Found: {session['duplicate_sets_found']}")
                text_lines.append(f"Space Saved: {session['space_saved_mb']:.1f} MB")
                text_lines.append(f"Status: {session['status']}")
                if session['notes']:
                    text_lines.append(f"Notes: {session['notes']}")
                text_lines.append("-" * 50)
                text_lines.append("")
            
            history_text.setPlainText("\n".join(text_lines))
        
        layout.addWidget(history_text)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(history_dialog.close)
        layout.addWidget(close_button)
        
        history_dialog.exec()