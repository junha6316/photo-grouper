"""
Photo Select Session Dialog for side-by-side image comparisons.

Provides a user-friendly comparison interface for deduplication sessions.
"""

import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
     QWidget, QProgressBar, QMessageBox, QFrame,
     QFileDialog
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QKeySequence, QShortcut

import tempfile


from core.select_session import SelectSession, ComparisonResult
from ui.components.zoomable_image_label import ZoomableImageLabel

class ComparisonWidget(QWidget):
    """Widget for side-by-side image comparison."""
    
    def __init__(self):
        super().__init__()
        self.champion_label = None
        self.challenger_label = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the comparison UI."""
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side
        left_layout = QVBoxLayout()
        
        left_title = QLabel("Image A")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2e7d32;
                padding: 8px;
                background-color: #e8f5e8;
                border-radius: 6px;
                border: 2px solid #4caf50;
            }
        """)
        left_layout.addWidget(left_title)
        
        self.left_label = ZoomableImageLabel()
        left_layout.addWidget(self.left_label, 1)
        
        self.left_filename = QLabel()
        self.left_filename.setAlignment(Qt.AlignCenter)
        self.left_filename.setStyleSheet("font-size: 12px; color: #666; padding: 5px;")
        self.left_filename.setWordWrap(True)
        left_layout.addWidget(self.left_filename)
        
        layout.addLayout(left_layout, 1)
        
        # VS separator
        vs_layout = QVBoxLayout()
        vs_layout.addStretch()
        
        vs_label = QLabel("VS")
        vs_label.setAlignment(Qt.AlignCenter)
        vs_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #ff6b35;
                background-color: #fff3e0;
                border: 2px solid #ff9800;
                border-radius: 25px;
                padding: 10px 20px;
                min-width: 60px;
            }
        """)
        vs_layout.addWidget(vs_label)
        vs_layout.addStretch()
        
        layout.addLayout(vs_layout, 0)
        
        # Right side
        right_layout = QVBoxLayout()
        
        right_title = QLabel("Image B")
        right_title.setAlignment(Qt.AlignCenter)
        right_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #c62828;
                padding: 8px;
                background-color: #ffebee;
                border-radius: 6px;
                border: 2px solid #f44336;
            }
        """)
        right_layout.addWidget(right_title)
        
        self.right_label = ZoomableImageLabel()
        right_layout.addWidget(self.right_label, 1)
        
        self.right_filename = QLabel()
        self.right_filename.setAlignment(Qt.AlignCenter)
        self.right_filename.setStyleSheet("font-size: 12px; color: #666; padding: 5px;")
        self.right_filename.setWordWrap(True)
        right_layout.addWidget(self.right_filename)
        
        layout.addLayout(right_layout, 1)
    
    def set_images(self, image_a_path: str, image_b_path: str):
        """Set the images for comparison."""
        self.left_label.set_image(image_a_path)
        self.right_label.set_image(image_b_path)
        
        # Update filenames
        image_a_filename = os.path.basename(image_a_path)
        image_b_filename = os.path.basename(image_b_path)
        
        self.left_filename.setText(image_a_filename)
        self.right_filename.setText(image_b_filename)

class SelectSessionDialog(QDialog):
    """Main dialog for photo deduplication sessions."""
    
    session_completed = Signal(dict)  # Emitted when session is complete
    
    def __init__(self, session: SelectSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_session)
        self.auto_save_timer.start(30000)  # Auto-save every 30 seconds
        
        self.setWindowTitle(f"Photo Deduplication Session - {session.session_id}")
        self.setGeometry(100, 100, 1400, 900)
        
        # Track if session is paused
        self.is_paused = False
        
        self.init_ui()
        self.setup_keyboard_shortcuts()
        self.load_next_comparison()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Header with progress and session info
        self.create_header(layout)
        
        # Main comparison area
        self.comparison_widget = ComparisonWidget()
        layout.addWidget(self.comparison_widget, 1)
        
        # Action buttons
        self.create_action_buttons(layout)
        
        # Status bar
        self.create_status_bar(layout)
    
    def create_header(self, layout):
        """Create the header with progress and session info."""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        header_layout = QVBoxLayout(header_frame)
        
        # Session title
        title_layout = QHBoxLayout()
        
        session_title = QLabel(f"Deduplication Session: {self.session.session_id}")
        session_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        title_layout.addWidget(session_title)
        
        title_layout.addStretch()
        
        # Session controls
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        title_layout.addWidget(self.pause_button)
        
        save_button = QPushButton("Save Session")
        save_button.clicked.connect(self.save_session)
        save_button.setStyleSheet(self.pause_button.styleSheet())
        title_layout.addWidget(save_button)
        
        header_layout.addLayout(title_layout)
        
        # Progress bar and info
        progress_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar, 1)
        
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size: 12px; color: #666; margin-left: 10px;")
        progress_layout.addWidget(self.progress_label)
        
        header_layout.addLayout(progress_layout)
        
        # Current group info
        self.group_info_label = QLabel()
        self.group_info_label.setStyleSheet("font-size: 12px; color: #666; margin-top: 5px;")
        header_layout.addWidget(self.group_info_label)
        
        layout.addWidget(header_frame)
    
    def create_action_buttons(self, layout):
        """Create the action buttons."""
        button_frame = QFrame()
        button_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        
        button_layout = QHBoxLayout(button_frame)
        button_layout.setSpacing(15)
        
        # Left image wins button
        self.left_button = QPushButton("← Choose Left (A)")
        self.left_button.clicked.connect(lambda: self.make_comparison(ComparisonResult.LEFT_WINS))
        self.left_button.setStyleSheet("""
            QPushButton {
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #4caf50;
                border-radius: 8px;
                background-color: #e8f5e8;
                color: #2e7d32;
            }
            QPushButton:hover {
                background-color: #c8e6c9;
            }
            QPushButton:pressed {
                background-color: #a5d6a7;
            }
        """)
        button_layout.addWidget(self.left_button)
        
        # Right image wins button
        self.right_button = QPushButton("Choose Right (B) →")
        self.right_button.clicked.connect(lambda: self.make_comparison(ComparisonResult.RIGHT_WINS))
        self.right_button.setStyleSheet("""
            QPushButton {
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #f44336;
                border-radius: 8px;
                background-color: #ffebee;
                color: #c62828;
            }
            QPushButton:hover {
                background-color: #ffcdd2;
            }
            QPushButton:pressed {
                background-color: #ef9a9a;
            }
        """)
        button_layout.addWidget(self.right_button)
        
        button_layout.addStretch()
        
        # Utility buttons
        self.undo_button = QPushButton("↶ Undo")
        self.undo_button.clicked.connect(self.undo_comparison)
        self.undo_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 12px;
                border: 1px solid #ff9800;
                border-radius: 4px;
                background-color: #fff3e0;
                color: #f57c00;
            }
            QPushButton:hover {
                background-color: #ffe0b2;
            }
            QPushButton:disabled {
                color: #999;
                background-color: #f5f5f5;
                border-color: #ddd;
            }
        """)
        button_layout.addWidget(self.undo_button)
        
        self.skip_button = QPushButton("⏭️ Skip Group")
        self.skip_button.clicked.connect(self.skip_group)
        self.skip_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 12px;
                border: 1px solid #9c27b0;
                border-radius: 4px;
                background-color: #f3e5f5;
                color: #7b1fa2;
            }
            QPushButton:hover {
                background-color: #e1bee7;
            }
        """)
        button_layout.addWidget(self.skip_button)
        
        close_button = QPushButton("✕ Close")
        close_button.clicked.connect(self.close_session)
        close_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 12px;
                border: 1px solid #666;
                border-radius: 4px;
                background-color: #f8f9fa;
                color: #666;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        button_layout.addWidget(close_button)
        
        layout.addWidget(button_frame)
    
    def create_status_bar(self, layout):
        """Create the status bar."""
        self.status_label = QLabel("Ready for comparison")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                padding: 8px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.status_label)
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Left arrow or 'A' for left image wins
        left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        left_shortcut.activated.connect(lambda: self.make_comparison(ComparisonResult.LEFT_WINS))
        
        left_shortcut2 = QShortcut(QKeySequence(Qt.Key_A), self)
        left_shortcut2.activated.connect(lambda: self.make_comparison(ComparisonResult.LEFT_WINS))
        
        # Right arrow or 'B' for right image wins
        right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        right_shortcut.activated.connect(lambda: self.make_comparison(ComparisonResult.RIGHT_WINS))
        
        right_shortcut2 = QShortcut(QKeySequence(Qt.Key_B), self)
        right_shortcut2.activated.connect(lambda: self.make_comparison(ComparisonResult.RIGHT_WINS))
        
        # U for undo
        undo_shortcut = QShortcut(QKeySequence(Qt.Key_U), self)
        undo_shortcut.activated.connect(self.undo_comparison)
        
        # S for skip
        skip_shortcut = QShortcut(QKeySequence(Qt.Key_S), self)
        skip_shortcut.activated.connect(self.skip_group)
        
        # Escape to close
        escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        escape_shortcut.activated.connect(self.close_session)
    
    def load_next_comparison(self):
        """Load the next comparison or complete the session."""
        if self.is_paused:
            return
        
        comparison = self.session.get_current_comparison()
        
        if comparison is None:
            # Session complete
            self.session_complete()
            return
        
        image_a, image_b, group_index, total_groups = comparison
        
        # Update UI
        self.comparison_widget.set_images(image_a, image_b)
        self.update_progress()
        self.update_group_info(group_index, total_groups)
        self.update_button_states()
        
        # Update status
        self.status_label.setText(f"Choose the better image: {os.path.basename(image_a)} vs {os.path.basename(image_b)}")
    
    def make_comparison(self, result: ComparisonResult):
        """Process a comparison result."""
        if self.is_paused:
            return
        
        success = self.session.make_comparison(result)
        if success:
            if result == ComparisonResult.LEFT_WINS:
                self.status_label.setText("Left image chosen! Loading next comparison...")
            elif result == ComparisonResult.RIGHT_WINS:
                self.status_label.setText("Right image chosen! Loading next comparison...")
            elif result == ComparisonResult.SKIP:
                self.status_label.setText("Group skipped. Loading next comparison...")
            
            # Small delay to show the result before loading next
            QTimer.singleShot(500, self.load_next_comparison)
        else:
            self.status_label.setText("Error processing comparison.")
    
    def undo_comparison(self):
        """Undo the last comparison."""
        if self.is_paused:
            return
        
        success = self.session.undo_last_comparison()
        if success:
            self.status_label.setText("Last comparison undone.")
            self.load_next_comparison()
        else:
            self.status_label.setText("No comparisons to undo.")
    
    def skip_group(self):
        """Skip the current group."""
        if self.is_paused:
            return
        
        success = self.session.skip_current_group()
        if success:
            self.status_label.setText("Group skipped.")
            self.load_next_comparison()
        else:
            self.status_label.setText("No group to skip.")
    
    def toggle_pause(self):
        """Toggle session pause state."""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_button.setText("Resume")
            self.status_label.setText("Session paused.")
            self.left_button.setEnabled(False)
            self.right_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.skip_button.setEnabled(False)
        else:
            self.pause_button.setText("Pause")
            self.status_label.setText("Session resumed.")
            self.update_button_states()
            self.load_next_comparison()
    
    def update_progress(self):
        """Update the progress bar and label."""
        completed, total, percentage = self.session.get_progress()
        
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(completed)
        
        self.progress_label.setText(f"{completed}/{total} comparisons ({percentage:.1f}%)")
    
    def update_group_info(self, group_index: int, total_groups: int):
        """Update the group information label."""
        self.group_info_label.setText(f"Group {group_index + 1} of {total_groups}")
    
    def update_button_states(self):
        """Update button enabled states."""
        has_history = len(self.session.state.history) > 0
        has_comparison = self.session.get_current_comparison() is not None
        
        self.undo_button.setEnabled(has_history and not self.is_paused)
        self.left_button.setEnabled(has_comparison and not self.is_paused)
        self.right_button.setEnabled(has_comparison and not self.is_paused)
        self.skip_button.setEnabled(has_comparison and not self.is_paused)
    
    def session_complete(self):
        """Handle session completion."""
        results = self.session.get_results()
        
        # Show completion message
        msg = QMessageBox(self)
        msg.setWindowTitle("Session Complete")
        msg.setIcon(QMessageBox.Information)
        msg.setText("Deduplication session completed!")
        msg.setInformativeText(
            f"Processed {results['completed_groups']} groups\n"
            f"Found {len(results['winners'])} winners\n"
            f"Skipped {results['skipped_groups']} groups\n"
            f"Total comparisons: {results['completed_comparisons']}"
        )
        msg.exec()
        
        # Emit completion signal
        self.session_completed.emit(results)
        
        # Close dialog
        self.accept()
    
    def save_session(self):
        """Save the session to a file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            f"session_{self.session.session_id}.json",
            "JSON Files (*.json)"
        )
        
        if filename:
            success = self.session.save_session(filename)
            if success:
                self.status_label.setText(f"Session saved to {filename}")
            else:
                self.status_label.setText("Error saving session.")
    
    def auto_save_session(self):
        """Auto-save the session."""
        temp_dir = tempfile.gettempdir()
        filename = os.path.join(temp_dir, f"autosave_{self.session.session_id}.json")
        self.session.save_session(filename)
    
    def close_session(self):
        """Close the session with confirmation."""
        if not self.session.is_session_complete():
            reply = QMessageBox.question(
                self,
                "Close Session",
                "Session is not complete. Do you want to save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_session()
                self.accept()
            elif reply == QMessageBox.Discard:
                self.reject()
            # Cancel does nothing
        else:
            self.accept()
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.session.is_session_complete():
            event.accept()
        else:
            self.close_session()
            event.ignore()