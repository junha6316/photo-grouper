"""
Persistent thumbnail cache using SQLite for optimized image loading.
"""

import sqlite3
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict
from contextlib import contextmanager
from PIL import Image
import io


class ThumbnailCache:
    """SQLite-based cache for image thumbnails."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the thumbnail cache.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            cache_dir = Path.home() / ".photo_grouper"
            cache_dir.mkdir(exist_ok=True)
            db_path = cache_dir / "thumbnails.db"
        
        self.db_path = str(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create thumbnails table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS thumbnails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_mtime REAL NOT NULL,
                    thumbnail_size INTEGER NOT NULL,
                    thumbnail_data BLOB NOT NULL,
                    format TEXT DEFAULT 'JPEG',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_path, file_hash, thumbnail_size)
                )
            """)
            
            # Create indices for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_thumbnail_lookup 
                ON thumbnails(file_path, thumbnail_size)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_thumbnail_hash 
                ON thumbnails(file_hash, thumbnail_size)
            """)
            
            # Create cache metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def _get_file_info(self, file_path: str) -> Tuple[str, int, float]:
        """Get file hash, size, and modification time."""
        try:
            stat = os.stat(file_path)
            file_size = stat.st_size
            file_mtime = stat.st_mtime
            
            # Generate file hash based on path, size, and mtime for efficiency
            hash_string = f"{file_path}_{file_size}_{file_mtime}"
            file_hash = hashlib.md5(hash_string.encode()).hexdigest()
            
            return file_hash, file_size, file_mtime
        except OSError:
            raise ValueError(f"Cannot access file: {file_path}")
    
    def get_thumbnail(self, file_path: str, size: int) -> Optional[bytes]:
        """
        Retrieve thumbnail from cache.
        
        Args:
            file_path: Path to the image file
            size: Thumbnail size (max dimension)
            
        Returns:
            Cached thumbnail bytes or None if not found/invalid
        """
        try:
            file_hash, file_size, file_mtime = self._get_file_info(file_path)
        except ValueError:
            return None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT thumbnail_data, file_size, file_mtime
                FROM thumbnails 
                WHERE file_path = ? AND file_hash = ? AND thumbnail_size = ?
            """, (file_path, file_hash, size))
            
            result = cursor.fetchone()
            
            if result is None:
                return None
            
            # Verify file hasn't changed
            cached_size, cached_mtime = result['file_size'], result['file_mtime']
            if cached_size != file_size or abs(cached_mtime - file_mtime) > 1.0:
                # File has changed, remove invalid cache entry
                self.remove_thumbnail(file_path, size)
                return None
            
            return result['thumbnail_data']
    
    def save_thumbnail(self, file_path: str, thumbnail_data: bytes, 
                      size: int, format: str = 'JPEG') -> bool:
        """
        Save thumbnail to cache.
        
        Args:
            file_path: Path to the image file
            thumbnail_data: Thumbnail image bytes
            size: Thumbnail size (max dimension)
            format: Image format (JPEG or PNG)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            file_hash, file_size, file_mtime = self._get_file_info(file_path)
        except ValueError:
            return False
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO thumbnails 
                    (file_path, file_hash, file_size, file_mtime, 
                     thumbnail_size, thumbnail_data, format)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (file_path, file_hash, file_size, file_mtime,
                      size, thumbnail_data, format))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving thumbnail for {file_path}: {e}")
            return False
    
    def create_and_cache_thumbnail(self, file_path: str, size: int,
                                  quality: int = 85) -> Optional[bytes]:
        """
        Create a thumbnail and save it to cache.
        
        Args:
            file_path: Path to the image file
            size: Thumbnail size (max dimension)
            quality: JPEG quality (1-100)
            
        Returns:
            Thumbnail bytes or None if failed
        """
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if needed (handles RGBA, LA, P modes)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Create thumbnail maintaining aspect ratio
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                
                # Save to bytes buffer
                buffer = io.BytesIO()
                # Use JPEG for photos, PNG for images with transparency
                format = 'JPEG' if img.mode == 'RGB' else 'PNG'
                save_kwargs = {'format': format}
                if format == 'JPEG':
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                
                img.save(buffer, **save_kwargs)
                thumbnail_data = buffer.getvalue()
                
                # Cache the thumbnail
                self.save_thumbnail(file_path, thumbnail_data, size, format)
                
                return thumbnail_data
                
        except Exception as e:
            print(f"Error creating thumbnail for {file_path}: {e}")
            return None
    
    def remove_thumbnail(self, file_path: str, size: Optional[int] = None) -> bool:
        """
        Remove thumbnail(s) from cache.
        
        Args:
            file_path: Path to the image file
            size: Specific thumbnail size to remove, or None for all sizes
            
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if size is not None:
                    cursor.execute("""
                        DELETE FROM thumbnails 
                        WHERE file_path = ? AND thumbnail_size = ?
                    """, (file_path, size))
                else:
                    cursor.execute("""
                        DELETE FROM thumbnails 
                        WHERE file_path = ?
                    """, (file_path,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total thumbnails
                cursor.execute("SELECT COUNT(*) as total FROM thumbnails")
                total = cursor.fetchone()['total']
                
                # Total size
                cursor.execute("SELECT SUM(LENGTH(thumbnail_data)) as total_size FROM thumbnails")
                total_size = cursor.fetchone()['total_size'] or 0
                
                # By size
                cursor.execute("""
                    SELECT thumbnail_size, COUNT(*) as count,
                           SUM(LENGTH(thumbnail_data)) as size
                    FROM thumbnails 
                    GROUP BY thumbnail_size
                    ORDER BY thumbnail_size
                """)
                
                by_size = {}
                for row in cursor.fetchall():
                    by_size[row['thumbnail_size']] = {
                        'count': row['count'],
                        'size_mb': (row['size'] or 0) / (1024 * 1024)
                    }
                
                return {
                    'total_thumbnails': total,
                    'total_size_mb': total_size / (1024 * 1024),
                    'by_size': by_size
                }
                
        except Exception:
            return {
                'total_thumbnails': 0,
                'total_size_mb': 0,
                'by_size': {}
            }
    
    def clear_cache(self, size: Optional[int] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            size: If specified, only clear entries for this thumbnail size
            
        Returns:
            Number of entries removed
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if size is not None:
                    cursor.execute("DELETE FROM thumbnails WHERE thumbnail_size = ?", (size,))
                else:
                    cursor.execute("DELETE FROM thumbnails")
                
                conn.commit()
                
                # Run VACUUM to reclaim space
                conn.execute("VACUUM")
                
                return cursor.rowcount
                
        except Exception:
            return 0
    
    def cleanup_invalid_entries(self) -> int:
        """Remove entries for files that no longer exist."""
        removed = 0
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT file_path FROM thumbnails")
                
                paths_to_remove = []
                for row in cursor.fetchall():
                    if not os.path.exists(row['file_path']):
                        paths_to_remove.append(row['file_path'])
                
                for path in paths_to_remove:
                    cursor.execute("DELETE FROM thumbnails WHERE file_path = ?", (path,))
                    removed += cursor.rowcount
                
                conn.commit()
                
                if removed > 0:
                    conn.execute("VACUUM")
                    
        except Exception:
            pass
        
        return removed