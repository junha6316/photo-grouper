"""
Session Launcher Dialog for configuring photo deduplication sessions.

Allows users to configure session parameters and start new sessions or load existing ones.
"""

import os
import tempfile
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QFileDialog, QMessageBox, QTextEdit, QComboBox, QListWidget,
    QListWidgetItem, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from typing import List, Dict, Any, Optional
import json
import glob

from core.select_session import SelectSession

class SessionLauncher(QDialog):
    """Dialog for configuring and launching deduplication sessions."""
    
    session_ready = Signal(SelectSession)  # Emitted when session is ready to start
    
    def __init__(self, groups: List[List[str]], embeddings: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.groups = groups
        self.embeddings = embeddings
        self.filtered_groups = []
        
        self.setWindowTitle("Start Deduplication Session")
        self.setGeometry(100, 100, 800, 700)
        
        self.init_ui()
        self.update_filtered_groups()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Configure Deduplication Session")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Session configuration
        self.create_session_config(layout)
        
        # Group filtering
        self.create_group_filtering(layout)
        
        # Preview section
        self.create_preview_section(layout)
        
        # Action buttons
        self.create_action_buttons(layout)
    
    def create_session_config(self, layout):
        """Create session configuration section."""
        config_group = QGroupBox("Session Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        config_layout = QVBoxLayout(config_group)
        
        # Similarity threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold:"))
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(80, 99)  # 0.80 to 0.99
        self.threshold_slider.setValue(90)  # Default 0.90
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider, 1)
        
        self.threshold_value = QLabel("0.90")
        self.threshold_value.setStyleSheet("font-weight: bold; color: #007acc; min-width: 40px;")
        threshold_layout.addWidget(self.threshold_value)
        
        config_layout.addLayout(threshold_layout)
        
        # Help text
        help_text = QLabel("Higher values = more similar images required for grouping")
        help_text.setStyleSheet("font-size: 11px; color: #666; margin-left: 20px;")
        config_layout.addWidget(help_text)
        
        # Minimum group size
        group_size_layout = QHBoxLayout()
        group_size_layout.addWidget(QLabel("Minimum Group Size:"))
        
        self.min_group_size = QSpinBox()
        self.min_group_size.setRange(2, 20)
        self.min_group_size.setValue(2)
        self.min_group_size.valueChanged.connect(self.update_filtered_groups)
        group_size_layout.addWidget(self.min_group_size)
        
        group_size_layout.addWidget(QLabel("images"))
        group_size_layout.addStretch()
        config_layout.addLayout(group_size_layout)
        
        # Auto-save options
        autosave_layout = QHBoxLayout()
        self.autosave_enabled = QCheckBox("Enable auto-save every")
        self.autosave_enabled.setChecked(True)
        autosave_layout.addWidget(self.autosave_enabled)
        
        self.autosave_interval = QSpinBox()
        self.autosave_interval.setRange(10, 300)
        self.autosave_interval.setValue(30)
        self.autosave_interval.setSuffix(" seconds")
        autosave_layout.addWidget(self.autosave_interval)
        
        autosave_layout.addStretch()
        config_layout.addLayout(autosave_layout)
        
        layout.addWidget(config_group)
    
    def create_group_filtering(self, layout):
        """Create group filtering section."""
        filter_group = QGroupBox("Group Filtering")
        filter_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        filter_layout = QVBoxLayout(filter_group)
        
        # Filter summary
        self.filter_summary = QLabel()
        self.filter_summary.setStyleSheet("font-size: 12px; color: #333; padding: 5px; background-color: #f8f9fa; border-radius: 4px;")
        filter_layout.addWidget(self.filter_summary)
        
        # Filter options
        options_layout = QHBoxLayout()
        
        self.include_large_groups = QCheckBox("Include large groups (>10 images)")
        self.include_large_groups.setChecked(True)
        self.include_large_groups.stateChanged.connect(self.update_filtered_groups)
        options_layout.addWidget(self.include_large_groups)
        
        options_layout.addStretch()
        filter_layout.addLayout(options_layout)
        
        layout.addWidget(filter_group)
    
    def create_preview_section(self, layout):
        """Create preview section."""
        preview_group = QGroupBox("Session Preview")
        preview_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        preview_layout = QVBoxLayout(preview_group)
        
        # Statistics
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("font-size: 12px; color: #333; padding: 8px; background-color: #e8f4fd; border-radius: 4px; border: 1px solid #bee5eb;")
        preview_layout.addWidget(self.stats_label)
        
        # Group list
        preview_layout.addWidget(QLabel("Groups to process:"))
        
        self.group_list = QListWidget()
        self.group_list.setMaximumHeight(120)
        self.group_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #fafafa;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
            }
        """)
        preview_layout.addWidget(self.group_list)
        
        layout.addWidget(preview_group)
    
    def create_action_buttons(self, layout):
        """Create action buttons."""
        # Load existing session section
        load_frame = QFrame()
        load_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        load_layout = QHBoxLayout(load_frame)
        
        load_layout.addWidget(QLabel("Or load existing session:"))
        
        load_button = QPushButton("Load Session...")
        load_button.clicked.connect(self.load_existing_session)
        load_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 12px;
                border: 1px solid #6c757d;
                border-radius: 4px;
                background-color: #f8f9fa;
                color: #495057;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        load_layout.addWidget(load_button)
        
        # Check for auto-saved sessions and add button if found
        self.check_autosaved_sessions()
        if hasattr(self, 'autosave_file'):
            autosave_button = QPushButton(f"Load Recent Auto-save ({self.autosave_time})")
            autosave_button.clicked.connect(lambda: self.load_session_file(self.autosave_file))
            autosave_button.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    font-size: 11px;
                    border: 1px solid #17a2b8;
                    border-radius: 4px;
                    background-color: #d1ecf1;
                    color: #0c5460;
                }
                QPushButton:hover {
                    background-color: #bee5eb;
                }
            """)
            load_layout.addWidget(autosave_button)
        
        load_layout.addStretch()
        
        layout.addWidget(load_frame)
        
        # Main action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                font-size: 14px;
                border: 1px solid #6c757d;
                border-radius: 6px;
                background-color: #f8f9fa;
                color: #495057;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        button_layout.addWidget(cancel_button)
        
        button_layout.addStretch()
        
        self.start_button = QPushButton("🚀 Start Session")
        self.start_button.clicked.connect(self.start_session)
        self.start_button.setStyleSheet("""
            QPushButton {
                padding: 12px 25px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #28a745;
                border-radius: 6px;
                background-color: #d4edda;
                color: #155724;
            }
            QPushButton:hover {
                background-color: #c3e6cb;
            }
            QPushButton:disabled {
                background-color: #f8f9fa;
                color: #6c757d;
                border-color: #dee2e6;
            }
        """)
        button_layout.addWidget(self.start_button)
        
        layout.addLayout(button_layout)
    
    def on_threshold_changed(self, value):
        """Handle threshold slider changes."""
        threshold = value / 100.0
        self.threshold_value.setText(f"{threshold:.2f}")
        self.update_filtered_groups()
    
    def update_filtered_groups(self):
        """Update the filtered groups based on current settings."""
        threshold = self.threshold_slider.value() / 100.0
        min_size = self.min_group_size.value()
        include_large = self.include_large_groups.isChecked()
        
        # Filter groups by size and similarity if we have embeddings
        self.filtered_groups = []
        
        for group in self.groups:
            if len(group) < min_size:
                continue
            
            if not include_large and len(group) > 10:
                continue
            
            # Check if group has high internal similarity
            if self.embeddings and len(group) > 1:
                group_similarity = self.calculate_group_similarity(group)
                if group_similarity >= threshold:
                    self.filtered_groups.append(group)
            else:
                self.filtered_groups.append(group)
        
        self.update_preview()
    
    def calculate_group_similarity(self, group: List[str]) -> float:
        """Calculate average internal similarity of a group."""
        if len(group) < 2:
            return 1.0
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Get embeddings for this group
            group_embeddings = []
            for path in group:
                if path in self.embeddings:
                    group_embeddings.append(self.embeddings[path])
            
            if len(group_embeddings) < 2:
                return 0.0
            
            # Calculate pairwise similarities
            embedding_matrix = np.array(group_embeddings)
            similarities = cosine_similarity(embedding_matrix)
            
            # Get upper triangle (excluding diagonal)
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            return float(np.mean(upper_triangle))
            
        except Exception as e:
            print(f"Error calculating group similarity: {e}")
            return 0.0
    
    def update_preview(self):
        """Update the preview section."""
        total_images = sum(len(group) for group in self.filtered_groups)
        total_comparisons = sum(len(group) - 1 for group in self.filtered_groups)
        
        # Update statistics
        stats_text = (
            f"Groups to process: {len(self.filtered_groups)}\n"
            f"Total images: {total_images}\n"
            f"Estimated comparisons: {total_comparisons}\n"
            f"Estimated time: {self.estimate_session_time(total_comparisons)}"
        )
        self.stats_label.setText(stats_text)
        
        # Update filter summary
        threshold = self.threshold_slider.value() / 100.0
        filter_text = f"Showing groups with ≥{len(self.filtered_groups)} images and ≥{threshold:.2f} similarity"
        self.filter_summary.setText(filter_text)
        
        # Update group list
        self.group_list.clear()
        for i, group in enumerate(self.filtered_groups[:20]):  # Show first 20 groups
            group_text = f"Group {i+1}: {len(group)} images"
            if group:
                first_file = os.path.basename(group[0])
                group_text += f" (e.g., {first_file})"
            
            item = QListWidgetItem(group_text)
            self.group_list.addItem(item)
        
        if len(self.filtered_groups) > 20:
            item = QListWidgetItem(f"... and {len(self.filtered_groups) - 20} more groups")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.group_list.addItem(item)
        
        # Enable/disable start button
        self.start_button.setEnabled(len(self.filtered_groups) > 0)
    
    def estimate_session_time(self, comparisons: int) -> str:
        """Estimate session completion time."""
        # Assume 10 seconds per comparison on average
        seconds = comparisons * 10
        
        if seconds < 60:
            return f"~{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"~{minutes} minutes"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"~{hours}h {minutes}m"
    
    def check_autosaved_sessions(self):
        """Check for auto-saved sessions and offer to load them."""
        temp_dir = tempfile.gettempdir()
        autosave_pattern = os.path.join(temp_dir, "autosave_session_*.json")
        autosave_files = glob.glob(autosave_pattern)
        
        if autosave_files:
            # Sort by modification time (newest first)
            autosave_files.sort(key=os.path.getmtime, reverse=True)
            
            # Show info about most recent autosave
            recent_file = autosave_files[0]
            mod_time = os.path.getmtime(recent_file)
            import time
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mod_time))
            
            # Store autosave info for later use instead of trying to add button immediately
            self.autosave_file = recent_file
            self.autosave_time = time_str
    
    def start_session(self):
        """Start a new deduplication session."""
        if not self.filtered_groups:
            QMessageBox.warning(self, "No Groups", "No groups meet the current filtering criteria.")
            return
        
        threshold = self.threshold_slider.value() / 100.0
        
        # Create new session
        session = SelectSession(self.filtered_groups, threshold)
        
        # Emit signal
        self.session_ready.emit(session)
        self.accept()
    
    def load_existing_session(self):
        """Load an existing session from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            "",
            "JSON Files (*.json)"
        )
        
        if filename:
            self.load_session_file(filename)
    
    def load_session_file(self, filename: str):
        """Load a session from a specific file."""
        session = SelectSession.load_session(filename)
        
        if session:
            reply = QMessageBox.question(
                self,
                "Load Session",
                f"Load session '{session.session_id}'?\n\n"
                f"Progress: {session.get_progress()[0]}/{session.get_progress()[1]} comparisons\n"
                f"Groups: {len(session.state.groups)}\n"
                f"Created: {session.state.created_at}",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.session_ready.emit(session)
                self.accept()
        else:
            QMessageBox.warning(self, "Load Error", "Failed to load session file.")
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        event.accept()