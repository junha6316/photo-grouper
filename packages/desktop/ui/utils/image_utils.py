"""
Utility functions for image loading and processing.
"""

import io
from typing import Optional, Tuple
from PIL import Image
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from infra.thumbnail_cache import ThumbnailCache

# Global thumbnail cache instance
_thumbnail_cache = None

def get_thumbnail_cache() -> ThumbnailCache:
    """Get the global thumbnail cache instance."""
    global _thumbnail_cache
    if _thumbnail_cache is None:
        _thumbnail_cache = ThumbnailCache()
    return _thumbnail_cache


def load_image_as_pixmap(image_path: str, max_size: Optional[int] = None,
                         maintain_aspect: bool = True, use_cache: bool = True) -> Optional[QPixmap]:
    """
    Load an image file and convert it to QPixmap with optional resizing.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width/height) for the image.
            If None, loads at original size
        maintain_aspect: Whether to maintain aspect ratio when resizing
        use_cache: Whether to use thumbnail cache for small sizes

    Returns:
        QPixmap of the loaded image, or None if loading failed
    """
    # Try to use cached thumbnail for small sizes
    if use_cache and max_size and max_size <= 256:
        cache = get_thumbnail_cache()
        
        # Check if thumbnail exists in cache
        thumbnail_data = cache.get_thumbnail(image_path, max_size)
        
        if thumbnail_data is None:
            # Create and cache new thumbnail
            thumbnail_data = cache.create_and_cache_thumbnail(image_path, max_size)
        
        if thumbnail_data:
            pixmap = QPixmap()
            pixmap.loadFromData(thumbnail_data)
            if not pixmap.isNull():
                return pixmap
    
    # Fallback to direct loading for large sizes or if cache fails
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed (handles RGBA, LA, P modes)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            # Resize if max_size is specified
            if max_size:
                if maintain_aspect:
                    # Calculate proper size maintaining aspect ratio
                    img.thumbnail((max_size, max_size),
                                  Image.Resampling.LANCZOS)
                else:
                    # Resize to exact size with high quality
                    img = img.resize((max_size, max_size),
                                     Image.Resampling.LANCZOS)

            # Use JPEG for smaller file size and faster loading
            img_buffer = io.BytesIO()
            if img.mode == 'RGB':
                img.save(img_buffer, format='JPEG', quality=90, optimize=True)
                format_str = 'JPEG'
            else:
                img.save(img_buffer, format='PNG')
                format_str = 'PNG'

            # Load QPixmap directly from bytes
            pixmap = QPixmap()
            pixmap.loadFromData(img_buffer.getvalue(), format_str)

            return pixmap if not pixmap.isNull() else None

    except Exception as e:
        return None


def scale_pixmap_to_size(pixmap: QPixmap, size: int) -> QPixmap:
    """
    Scale a pixmap to fit within a square of the given size.

    Args:
        pixmap: The QPixmap to scale
        size: The maximum dimension (width/height) for the scaled pixmap

    Returns:
        Scaled QPixmap
    """
    # Don't scale if already smaller than target
    if pixmap.width() <= size and pixmap.height() <= size:
        return pixmap

    return pixmap.scaled(
        size, size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """
    Get the dimensions of an image without fully loading it.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) or None if failed
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None