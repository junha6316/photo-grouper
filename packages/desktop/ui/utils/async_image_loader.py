import queue
import time
from typing import Optional
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, Future

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap

from ui.utils.image_utils import load_image_as_pixmap


class ImageLoadResult:
    """Result of an image loading operation."""
    def __init__(self, success: bool, pixmap: Optional[QPixmap] = None,
                 error: Optional[str] = None, image_path: str = ""):
        self.success = success
        self.pixmap = pixmap
        self.error = error
        self.image_path = image_path


class ImageLoadRequest:
    """Request for loading an image."""
    def __init__(self, image_path: str, size: int, priority: int = 0):
        self.image_path = image_path
        self.size = size
        self.priority = priority
        self.timestamp = time.time()
        
    def __lt__(self, other):
        # Higher priority first, then older requests first
        return (self.priority, -self.timestamp) > \
               (other.priority, -other.timestamp)


class AsyncImageLoaderThread(QThread):
    """Thread-based async image loader with parallel processing using
    ThreadPoolExecutor."""
    
    image_loaded = Signal(ImageLoadResult)
    
    def __init__(self, max_workers: int = 4):
        super().__init__()
        self.load_queue = queue.PriorityQueue()
        self.running = False
        self.cache = {}  # Simple in-memory cache
        self.cache_lock = Lock()
        self.max_workers = max_workers
        self.executor = None
        self.active_futures = {}  # Track active futures
        self.futures_lock = Lock()
        
    def run(self):
        """Run the image loading loop with parallel processing."""
        self.running = True
        
        # Initialize thread pool executor
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ImageLoader"
        )
        
        try:
            while self.running:
                try:
                    # Get next request with timeout
                    request = self.load_queue.get(timeout=1.0)
                    
                    # Check if we should stop
                    if not self.running or request.priority == -1000:
                        self.load_queue.task_done()
                        break
                    
                    # Submit to thread pool for parallel processing
                    future = self.executor.submit(
                        self._load_image_sync, request
                    )
                    
                    # Track the future and add callback
                    with self.futures_lock:
                        self.active_futures[future] = request
                    
                    future.add_done_callback(self._on_image_loaded_callback)
                    
                    # Mark task as done
                    self.load_queue.task_done()
                    
                except queue.Empty:
                    # No requests in queue, continue
                    continue
                except Exception as e:
                    print(f"Error in image loader thread: {e}")
        finally:
            # Clean up executor
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
    
    def _on_image_loaded_callback(self, future: Future):
        """Callback when an image loading task completes."""
        try:
            # Get the result from the future
            result = future.result()
            
            # Remove from active futures
            with self.futures_lock:
                if future in self.active_futures:
                    del self.active_futures[future]
            
            # Emit signal to main thread (this is thread-safe for Qt signals)
            self.image_loaded.emit(result)
            
        except Exception as e:
            # Handle any exceptions that occurred during image loading
            with self.futures_lock:
                request = self.active_futures.pop(future, None)
            
            error_path = request.image_path if request else "unknown"
            error_result = ImageLoadResult(False, None, f"Loading failed: {str(e)}", error_path)
            self.image_loaded.emit(error_result)
    
    def _load_image_sync(self, request: ImageLoadRequest) -> ImageLoadResult:
        """
        Synchronously load and process an image.
        This runs in the background thread to avoid blocking UI.
        """
        cache_key = f"{request.image_path}_{request.size}"
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        try:
            # Method 1: Try PIL thumbnail creation
            pixmap = self._create_pil_thumbnail(request.image_path, request.size)
            
            # Method 2: Fallback to direct QPixmap loading
            if not pixmap or pixmap.isNull():
                pixmap = self._create_direct_thumbnail(request.image_path, request.size)
            
            if pixmap and not pixmap.isNull():
                result = ImageLoadResult(True, pixmap, None, request.image_path)
            else:
                result = ImageLoadResult(False, None, "Failed to load image", request.image_path)
            
            # Cache the result
            with self.cache_lock:
                # Increased cache size for better performance
                if len(self.cache) < 500:  # Limit cache size
                    self.cache[cache_key] = result
                elif len(self.cache) >= 500:
                    # LRU eviction: remove oldest entry
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            error_msg = f"Error loading {request.image_path}: {str(e)}"
            return ImageLoadResult(False, None, error_msg, request.image_path)
    
    def _create_pil_thumbnail(self, image_path: str, size: int) -> Optional[QPixmap]:
        """Create thumbnail using PIL with caching."""
        # Use the centralized image loading utility with caching enabled
        return load_image_as_pixmap(image_path, max_size=size, use_cache=True)
    
    def _create_direct_thumbnail(self, image_path: str, size: int) -> Optional[QPixmap]:
        """Create thumbnail using direct QPixmap loading."""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                return pixmap.scaled(
                    size, size, 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
        except Exception:
            pass
        return None
    
    def get_cached_image(self, image_path: str, size: int) -> Optional[ImageLoadResult]:
        """
        Check if an image is already in cache and return it immediately.
        Returns None if not cached.
        This method is thread-safe and can be called from the main thread.
        """
        cache_key = f"{image_path}_{size}"
        with self.cache_lock:
            return self.cache.get(cache_key, None)
    
    def load_image(self, image_path: str, size: int = 150, priority: int = 0):
        """Queue an image for async loading."""
        if self.running:
            request = ImageLoadRequest(image_path, size, priority)
            self.load_queue.put(request)
    
    def clear_cache(self):
        """Clear the image cache."""
        with self.cache_lock:
            self.cache.clear()
    
    def get_load_status(self) -> dict:
        """Get current loading status for debugging."""
        with self.futures_lock:
            active_count = len(self.active_futures)
        
        queue_size = self.load_queue.qsize()
        cache_size = len(self.cache)
        
        return {
            'active_loads': active_count,
            'queue_size': queue_size,
            'cache_size': cache_size,
            'max_workers': self.max_workers,
            'running': self.running
        }
    
    def stop(self):
        """Stop the async loader thread and cleanup resources."""
        self.running = False
        
        # Cancel any pending futures
        if self.executor:
            with self.futures_lock:
                for future in list(self.active_futures.keys()):
                    future.cancel()
                self.active_futures.clear()
        
        # Add a dummy request to wake up the thread
        try:
            self.load_image("", 0, -1000)  # Special priority to signal stop
        except:
            pass
            
        # Wait for thread to finish
        self.quit()
        self.wait(5000)  # Wait up to 5 seconds for cleanup


# Global async image loader instance
_async_loader = None

def get_async_loader() -> AsyncImageLoaderThread:
    """Get the global async image loader instance."""
    global _async_loader
    if _async_loader is None:
        # Use more workers for better parallel performance
        # Adjust based on system capabilities (CPU cores)
        import os
        # Increased worker count for better throughput
        max_workers = min(12, (os.cpu_count() or 2) * 2)
        _async_loader = AsyncImageLoaderThread(max_workers=max_workers)
        _async_loader.start()
    return _async_loader

def cleanup_async_loader():
    """Clean up the global async loader."""
    global _async_loader
    if _async_loader is not None:
        _async_loader.stop()
        _async_loader = None