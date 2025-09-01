import os
from pathlib import Path
from typing import List, Set

class ImageScanner:
    SUPPORTED_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
        '.webp', '.heic', '.heif'
    }
    
    def __init__(self):
        pass
    
    def scan_images(self, folder_path: str) -> List[str]:
        """
        Scan a folder for supported image files.
        
        Args:
            folder_path: Path to the folder to scan
            
        Returns:
            List of absolute paths to image files
        """
        image_paths = []
        folder = Path(folder_path)
        
        if not folder.exists() or not folder.is_dir():
            return image_paths
        
        # Recursively find all image files
        for file_path in folder.rglob('*'):
            if file_path.is_file() and self._is_supported_image(file_path):
                image_paths.append(str(file_path.absolute()))
        
        return sorted(image_paths)
    
    def _is_supported_image(self, file_path: Path) -> bool:
        """Check if file has a supported image extension."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS