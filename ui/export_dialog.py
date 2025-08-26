"""
Dialog for exporting selected images to a folder.
"""

import os
import shutil
from pathlib import Path
from typing import List
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QRadioButton,
    QButtonGroup, QGroupBox, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from utils.preferences import get_preferences


class ExportWorker(QThread):
    """Worker thread for exporting images."""
    
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(int, int)  # successful, failed
    error = Signal(str)
    
    def __init__(self, images: List[str], destination: str, 
                 copy_mode: bool = True, organize_by_date: bool = False):
        super().__init__()
        self.images = images
        self.destination = destination
        self.copy_mode = copy_mode  # True for copy, False for move
        self.organize_by_date = organize_by_date
        self.should_stop = False
    
    def run(self):
        """Execute the export operation."""
        successful = 0
        failed = 0
        
        try:
            # Create destination directory if it doesn't exist
            Path(self.destination).mkdir(parents=True, exist_ok=True)
            
            total = len(self.images)
            
            for i, image_path in enumerate(self.images):
                if self.should_stop:
                    break
                
                try:
                    source = Path(image_path)
                    if not source.exists():
                        self.status.emit(f"Skipping missing file: {source.name}")
                        failed += 1
                        continue
                    
                    # Determine destination path
                    if self.organize_by_date:
                        # Get file modification date
                        mtime = datetime.fromtimestamp(source.stat().st_mtime)
                        date_folder = mtime.strftime("%Y-%m-%d")
                        dest_dir = Path(self.destination) / date_folder
                        dest_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        dest_dir = Path(self.destination)
                    
                    dest_path = dest_dir / source.name
                    
                    # Handle naming conflicts
                    if dest_path.exists():
                        base = dest_path.stem
                        ext = dest_path.suffix
                        counter = 1
                        while dest_path.exists():
                            dest_path = dest_dir / f"{base}_{counter}{ext}"
                            counter += 1
                    
                    # Copy or move the file
                    self.status.emit(f"{'Copying' if self.copy_mode else 'Moving'}: {source.name}")
                    
                    if self.copy_mode:
                        shutil.copy2(source, dest_path)
                    else:
                        shutil.move(str(source), str(dest_path))
                    
                    successful += 1
                    
                except Exception as e:
                    self.error.emit(f"Error processing {image_path}: {str(e)}")
                    failed += 1
                
                # Update progress
                progress = int((i + 1) / total * 100)
                self.progress.emit(progress)
            
            self.finished.emit(successful, failed)
            
        except Exception as e:
            self.error.emit(f"Export failed: {str(e)}")
            self.finished.emit(successful, failed)
    
    def stop(self):
        """Stop the export operation."""
        self.should_stop = True


class ExportDialog(QDialog):
    """Dialog for exporting selected images to a folder."""
    
    def __init__(self, selected_images: List[str], parent=None):
        super().__init__(parent)
        self.selected_images = selected_images
        self.worker = None
        self.prefs = get_preferences()
        
        self.setWindowTitle("Export Selected Images")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel(f"Export {len(self.selected_images)} selected images")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(info_label)
        
        # Destination folder selection
        dest_group = QGroupBox("Destination")
        dest_layout = QHBoxLayout()
        
        self.dest_input = QLineEdit()
        # Use last export path from preferences
        last_path = self.prefs.get('export.last_export_path', str(Path.home() / "Pictures"))
        default_dest = str(Path(last_path) / f"PhotoGrouper_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.dest_input.setText(default_dest)
        dest_layout.addWidget(self.dest_input)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_destination)
        dest_layout.addWidget(browse_btn)
        
        dest_group.setLayout(dest_layout)
        layout.addWidget(dest_group)
        
        # Export options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # Copy vs Move (load from preferences)
        copy_mode = self.prefs.get('export.copy_mode', True)
        self.copy_radio = QRadioButton("Copy images (keep originals)")
        self.copy_radio.setChecked(copy_mode)
        self.move_radio = QRadioButton("Move images (remove originals)")
        self.move_radio.setChecked(not copy_mode)
        
        mode_group = QButtonGroup()
        mode_group.addButton(self.copy_radio)
        mode_group.addButton(self.move_radio)
        
        options_layout.addWidget(self.copy_radio)
        options_layout.addWidget(self.move_radio)
        
        # Organization options (load from preferences)
        self.organize_by_date = QCheckBox("Organize by date (create date subfolders)")
        self.organize_by_date.setChecked(self.prefs.get('export.organize_by_date', False))
        options_layout.addWidget(self.organize_by_date)
        
        # Create shortcuts option
        self.create_shortcuts = QCheckBox("Create shortcuts/links instead of copying (macOS/Linux only)")
        self.create_shortcuts.setEnabled(os.name != 'nt')  # Disable on Windows
        if os.name == 'nt':
            self.create_shortcuts.setToolTip("Symbolic links are not fully supported on Windows")
        options_layout.addWidget(self.create_shortcuts)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Progress section
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.start_export)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                padding: 8px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0051D5;
            }
        """)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_export)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.export_btn)
        
        layout.addWidget(QLabel())  # Spacer
        layout.addLayout(button_layout)
    
    def browse_destination(self):
        """Browse for destination folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder",
            str(Path.home() / "Pictures"),
            QFileDialog.ShowDirsOnly
        )
        
        if folder:
            # Create a subfolder for the export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_folder = str(Path(folder) / f"PhotoGrouper_Export_{timestamp}")
            self.dest_input.setText(export_folder)
    
    def start_export(self):
        """Start the export operation."""
        destination = self.dest_input.text().strip()
        
        # Save preferences
        self.prefs.set('export.copy_mode', self.copy_radio.isChecked())
        self.prefs.set('export.organize_by_date', self.organize_by_date.isChecked())
        # Save parent directory of destination as last export path
        dest_parent = str(Path(destination).parent)
        self.prefs.set('export.last_export_path', dest_parent)
        
        if not destination:
            QMessageBox.warning(self, "Warning", "Please select a destination folder.")
            return
        
        # Check if destination exists and ask for confirmation
        dest_path = Path(destination)
        if dest_path.exists() and list(dest_path.iterdir()):
            reply = QMessageBox.question(
                self, 
                "Folder Exists",
                f"The folder '{dest_path.name}' already exists and is not empty.\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # Disable controls during export
        self.export_btn.setEnabled(False)
        self.dest_input.setEnabled(False)
        self.copy_radio.setEnabled(False)
        self.move_radio.setEnabled(False)
        self.organize_by_date.setEnabled(False)
        self.create_shortcuts.setEnabled(False)
        
        # Show progress
        self.progress_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Handle shortcuts/links
        if self.create_shortcuts.isChecked():
            self.create_symbolic_links(destination)
        else:
            # Start worker thread
            copy_mode = self.copy_radio.isChecked()
            organize = self.organize_by_date.isChecked()
            
            self.worker = ExportWorker(
                self.selected_images,
                destination,
                copy_mode,
                organize
            )
            
            self.worker.progress.connect(self.update_progress)
            self.worker.status.connect(self.update_status)
            self.worker.finished.connect(self.export_finished)
            self.worker.error.connect(self.show_error)
            
            self.worker.start()
    
    def create_symbolic_links(self, destination: str):
        """Create symbolic links instead of copying files."""
        try:
            Path(destination).mkdir(parents=True, exist_ok=True)
            
            successful = 0
            failed = 0
            
            for i, image_path in enumerate(self.selected_images):
                try:
                    source = Path(image_path).resolve()
                    dest = Path(destination) / source.name
                    
                    # Handle naming conflicts
                    if dest.exists():
                        base = dest.stem
                        ext = dest.suffix
                        counter = 1
                        while dest.exists():
                            dest = Path(destination) / f"{base}_{counter}{ext}"
                            counter += 1
                    
                    # Create symbolic link
                    dest.symlink_to(source)
                    successful += 1
                    
                except Exception as e:
                    print(f"Failed to create link for {image_path}: {e}")
                    failed += 1
                
                # Update progress
                progress = int((i + 1) / len(self.selected_images) * 100)
                self.progress_bar.setValue(progress)
            
            self.export_finished(successful, failed)
            
        except Exception as e:
            self.show_error(f"Failed to create symbolic links: {str(e)}")
    
    def update_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)
    
    def update_status(self, message: str):
        """Update status label."""
        self.progress_label.setText(message)
    
    def show_error(self, message: str):
        """Show error message."""
        QMessageBox.critical(self, "Error", message)
    
    def export_finished(self, successful: int, failed: int):
        """Handle export completion."""
        self.progress_bar.setValue(100)
        
        # Show result with option to open folder
        message = f"Export completed!\n\n"
        message += f"✓ {successful} images exported successfully"
        if failed > 0:
            message += f"\n✗ {failed} images failed"
        
        # Create custom message box with option to open folder
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Export Complete")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        
        # If export was successful, ask if user wants to open the folder
        if successful > 0:
            dest_path = Path(self.dest_input.text())
            if dest_path.exists():
                msg_box.setInformativeText("내보내기 폴더를 열까요?\nWould you like to open the destination folder?")
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg_box.setDefaultButton(QMessageBox.Yes)
        else:
            msg_box.setStandardButtons(QMessageBox.Ok)
        
        result = msg_box.exec()
        
        # Open folder if user clicked Yes
        if successful > 0 and result == QMessageBox.Yes:
            dest_path = Path(self.dest_input.text())
            if dest_path.exists():
                self.open_folder(dest_path)
        
        self.accept()
    
    def open_folder(self, folder_path: Path):
        """Open the specified folder in the system file manager."""
        try:
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                subprocess.run(['open', str(folder_path)], check=True)
            elif system == 'Windows':
                os.startfile(str(folder_path))
            else:  # Linux and other Unix-like systems
                # Try different methods for Linux
                openers = ['xdg-open', 'gnome-open', 'kde-open']
                opened = False
                
                for opener in openers:
                    try:
                        subprocess.run([opener, str(folder_path)], check=True, 
                                     stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                        opened = True
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                
                if not opened:
                    # If no opener worked, show error message
                    QMessageBox.warning(self, "Cannot Open Folder", 
                                       f"Could not open folder automatically.\n"
                                       f"Please navigate to:\n{folder_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open folder: {e}")
    
    def cancel_export(self):
        """Cancel the export operation."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        self.reject()