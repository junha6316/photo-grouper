#!/usr/bin/env python3
"""
Photo Grouper - Desktop application for grouping similar photos using ML embeddings.
"""

import sys
from PySide6.QtWidgets import QApplication
from windows.main_window import MainWindow
from ui.utils.async_image_loader import cleanup_async_loader

def main():
    """Main entry point for the Photo Grouper application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Photo Grouper")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start the application event loop
    try:
        sys.exit(app.exec())
    finally:
        # Clean up async loader on app exit
        cleanup_async_loader()

if __name__ == "__main__":
    main()