"""
Image preloading utilities for optimized viewport-based loading.
"""

from typing import List, Set, Optional
from PySide6.QtCore import QObject, Signal, QTimer
from ui.utils.async_image_loader import get_async_loader


class ViewportImagePreloader(QObject):
    """
    Manages viewport-based image preloading with priority queue.
    Preloads visible images first, then nearby images.
    """
    
    preload_started = Signal(int)  # Emits number of images being preloaded
    preload_completed = Signal(int)  # Emits number of images preloaded
    
    def __init__(self, thumbnail_size: int = 150, preload_margin: int = 2):
        """
        Initialize the viewport preloader.
        
        Args:
            thumbnail_size: Size of thumbnails to preload
            preload_margin: Number of rows/columns to preload beyond viewport
        """
        super().__init__()
        self.thumbnail_size = thumbnail_size
        self.preload_margin = preload_margin
        self.loader = get_async_loader()
        self.preloaded_images: Set[str] = set()
        self.loading_images: Set[str] = set()
        
        # Batch preloading timer
        self.preload_timer = QTimer()
        self.preload_timer.setSingleShot(True)
        self.preload_timer.timeout.connect(self._execute_preload)
        self.pending_preloads: List[tuple] = []
    
    def preload_viewport_images(self, visible_images: List[str], 
                               nearby_images: Optional[List[str]] = None):
        """
        Preload images based on viewport visibility.
        
        Args:
            visible_images: List of currently visible image paths
            nearby_images: List of nearby (off-screen) image paths
        """
        # Clear pending preloads
        self.pending_preloads.clear()
        
        # High priority: visible images not yet loaded
        for image_path in visible_images:
            if image_path not in self.preloaded_images and \
               image_path not in self.loading_images:
                self.pending_preloads.append((image_path, 10))  # Priority 10
        
        # Medium priority: nearby images
        if nearby_images:
            for image_path in nearby_images:
                if image_path not in self.preloaded_images and \
                   image_path not in self.loading_images:
                    self.pending_preloads.append((image_path, 5))  # Priority 5
        
        # Start batch loading with small delay to debounce rapid scrolling
        if self.pending_preloads:
            self.preload_timer.stop()
            self.preload_timer.start(50)  # 50ms delay
    
    def _execute_preload(self):
        """Execute the actual preloading."""
        if not self.pending_preloads:
            return
        
        # Emit signal about preload starting
        self.preload_started.emit(len(self.pending_preloads))
        
        # Submit all pending preloads
        for image_path, priority in self.pending_preloads:
            self.loading_images.add(image_path)
            
            # Check cache first
            cached = self.loader.get_cached_image(image_path, self.thumbnail_size)
            if cached:
                self.preloaded_images.add(image_path)
                self.loading_images.discard(image_path)
            else:
                # Queue for loading with priority
                self.loader.load_image(image_path, self.thumbnail_size, priority)
        
        # Connect to completion signal if not already connected
        try:
            self.loader.image_loaded.disconnect(self._on_image_loaded)
        except:
            pass
        self.loader.image_loaded.connect(self._on_image_loaded)
        
        self.pending_preloads.clear()
    
    def _on_image_loaded(self, result):
        """Handle image load completion."""
        if result.image_path in self.loading_images:
            self.loading_images.discard(result.image_path)
            if result.success:
                self.preloaded_images.add(result.image_path)
            
            # Emit completion if all done
            if not self.loading_images:
                self.preload_completed.emit(len(self.preloaded_images))
    
    def clear_preload_cache(self):
        """Clear the preload tracking (not the actual image cache)."""
        self.preloaded_images.clear()
        self.loading_images.clear()
        self.pending_preloads.clear()
        self.preload_timer.stop()
    
    def get_preload_stats(self) -> dict:
        """Get preloading statistics."""
        return {
            'preloaded': len(self.preloaded_images),
            'loading': len(self.loading_images),
            'pending': len(self.pending_preloads),
            'cache_stats': self.loader.get_load_status()
        }