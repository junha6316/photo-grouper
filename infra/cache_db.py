import sqlite3
import numpy as np
import json
import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from contextlib import contextmanager

class EmbeddingCache:
    """SQLite-based cache for image embeddings."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the embedding cache.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Default to user's home directory
            cache_dir = Path.home() / ".photo_grouper"
            cache_dir.mkdir(exist_ok=True)
            db_path = cache_dir / "embeddings.db"
        
        self.db_path = str(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_mtime REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding_dims INTEGER NOT NULL,
                    embedding_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_path, file_hash, model_name)
                )
            """)
            
            # Create index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path_hash 
                ON embeddings(file_path, file_hash, model_name)
            """)
            
            # Create metadata table for cache info
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create deduplication sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deduplication_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    similarity_threshold REAL NOT NULL,
                    total_images_processed INTEGER NOT NULL,
                    duplicate_sets_found INTEGER NOT NULL,
                    space_saved_mb REAL DEFAULT 0,
                    status TEXT DEFAULT 'completed',
                    notes TEXT
                )
            """)
            
            # Create deduplication files table to track processed files
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deduplication_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    duplicate_group_id INTEGER,
                    is_keeper BOOLEAN DEFAULT 0,
                    action_taken TEXT,  -- 'kept', 'deleted', 'moved', 'skipped'
                    moved_to_path TEXT,
                    similarity_score REAL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES deduplication_sessions(session_id)
                )
            """)
            
            # Create indices for deduplication tables
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_dedup_session_id 
                ON deduplication_files(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_dedup_file_path 
                ON deduplication_files(file_path)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_dedup_file_hash 
                ON deduplication_files(file_hash)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
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
    
    def get_embedding(self, file_path: str, model_name: str = "vgg16_pca") -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.
        
        Args:
            file_path: Path to the image file
            model_name: Name of the model used for embedding
            
        Returns:
            Cached embedding array or None if not found/invalid
        """
        try:
            file_hash, file_size, file_mtime = self._get_file_info(file_path)
        except ValueError:
            return None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT embedding_data, embedding_dims, file_size, file_mtime
                FROM embeddings 
                WHERE file_path = ? AND file_hash = ? AND model_name = ?
            """, (file_path, file_hash, model_name))
            
            result = cursor.fetchone()
            
            if result is None:
                return None
            
            # Verify file hasn't changed
            cached_size, cached_mtime = result['file_size'], result['file_mtime']
            if cached_size != file_size or abs(cached_mtime - file_mtime) > 1.0:
                # File has changed, remove invalid cache entry
                self.remove_embedding(file_path, model_name)
                return None
            
            # Deserialize embedding
            try:
                embedding_dims = result['embedding_dims']
                embedding_bytes = result['embedding_data']
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                if len(embedding) != embedding_dims:
                    # Corrupted data
                    self.remove_embedding(file_path, model_name)
                    return None
                
                return embedding
                
            except Exception:
                # Corrupted data
                self.remove_embedding(file_path, model_name)
                return None
    
    def save_embedding(self, file_path: str, embedding: np.ndarray, 
                      model_name: str = "vgg16_pca") -> bool:
        """
        Save embedding to cache.
        
        Args:
            file_path: Path to the image file
            embedding: Embedding array to cache
            model_name: Name of the model used for embedding
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            file_hash, file_size, file_mtime = self._get_file_info(file_path)
        except ValueError:
            return False
        
        try:
            # Convert embedding to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()
            embedding_dims = len(embedding)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (file_path, file_hash, file_size, file_mtime, model_name, 
                     embedding_dims, embedding_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (file_path, file_hash, file_size, file_mtime, model_name,
                      embedding_dims, embedding_bytes))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving embedding for {file_path}: {e}")
            return False
    
    def remove_embedding(self, file_path: str, model_name: str = "vgg16_pca") -> bool:
        """Remove embedding from cache."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM embeddings 
                    WHERE file_path = ? AND model_name = ?
                """, (file_path, model_name))
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) as total FROM embeddings")
                total = cursor.fetchone()['total']
                
                cursor.execute("""
                    SELECT model_name, COUNT(*) as count 
                    FROM embeddings 
                    GROUP BY model_name
                """)
                by_model = {row['model_name']: row['count'] for row in cursor.fetchall()}
                
                return {
                    'total_embeddings': total,
                    'by_model': by_model
                }
        except Exception:
            return {'total_embeddings': 0, 'by_model': {}}
    
    def clear_cache(self, model_name: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            model_name: If specified, only clear entries for this model
            
        Returns:
            Number of entries removed
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if model_name:
                    cursor.execute("DELETE FROM embeddings WHERE model_name = ?", (model_name,))
                else:
                    cursor.execute("DELETE FROM embeddings")
                
                conn.commit()
                return cursor.rowcount
        except Exception:
            return 0
    
    def cleanup_invalid_entries(self) -> int:
        """Remove entries for files that no longer exist."""
        removed = 0
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, file_path FROM embeddings")
                
                for row in cursor.fetchall():
                    if not os.path.exists(row['file_path']):
                        cursor.execute("DELETE FROM embeddings WHERE id = ?", (row['id'],))
                        removed += 1
                
                conn.commit()
        except Exception:
            pass
        
        return removed
    
    # Deduplication session management methods
    
    def create_deduplication_session(self, similarity_threshold: float, 
                                    total_images: int, notes: str = None) -> str:
        """
        Create a new deduplication session.
        
        Args:
            similarity_threshold: Threshold used for finding duplicates
            total_images: Total number of images being processed
            notes: Optional notes about the session
            
        Returns:
            Session ID string
        """
        import uuid
        from datetime import datetime
        
        session_id = f"dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO deduplication_sessions 
                    (session_id, similarity_threshold, total_images_processed, 
                     duplicate_sets_found, status, notes)
                    VALUES (?, ?, ?, 0, 'in_progress', ?)
                """, (session_id, similarity_threshold, total_images, notes))
                conn.commit()
                return session_id
        except Exception as e:
            print(f"Error creating deduplication session: {e}")
            return None
    
    def save_deduplication_file(self, session_id: str, file_path: str, 
                               file_hash: str, file_size: int,
                               duplicate_group_id: int = None,
                               is_keeper: bool = False,
                               action_taken: str = 'pending',
                               moved_to_path: str = None,
                               similarity_score: float = None) -> bool:
        """
        Save information about a file processed in deduplication.
        
        Args:
            session_id: Deduplication session ID
            file_path: Path to the image file
            file_hash: Hash of the file
            file_size: Size of the file in bytes
            duplicate_group_id: ID of the duplicate group this file belongs to
            is_keeper: Whether this file was chosen as the keeper
            action_taken: Action taken on the file ('kept', 'deleted', 'moved', 'skipped')
            moved_to_path: Path where file was moved to (if applicable)
            similarity_score: Similarity score within the duplicate group
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO deduplication_files 
                    (session_id, file_path, file_hash, file_size, 
                     duplicate_group_id, is_keeper, action_taken, 
                     moved_to_path, similarity_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, file_path, file_hash, file_size,
                      duplicate_group_id, is_keeper, action_taken,
                      moved_to_path, similarity_score))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving deduplication file: {e}")
            return False
    
    def update_deduplication_session(self, session_id: str, 
                                    duplicate_sets_found: int = None,
                                    space_saved_mb: float = None,
                                    status: str = None) -> bool:
        """
        Update deduplication session with results.
        
        Args:
            session_id: Deduplication session ID
            duplicate_sets_found: Number of duplicate sets found
            space_saved_mb: Space saved in MB
            status: Session status ('completed', 'failed', 'cancelled')
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                updates = []
                params = []
                
                if duplicate_sets_found is not None:
                    updates.append("duplicate_sets_found = ?")
                    params.append(duplicate_sets_found)
                
                if space_saved_mb is not None:
                    updates.append("space_saved_mb = ?")
                    params.append(space_saved_mb)
                
                if status is not None:
                    updates.append("status = ?")
                    params.append(status)
                
                if not updates:
                    return True
                
                params.append(session_id)
                query = f"UPDATE deduplication_sessions SET {', '.join(updates)} WHERE session_id = ?"
                
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error updating deduplication session: {e}")
            return False
    
    def get_deduplication_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent deduplication sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, created_at, similarity_threshold,
                           total_images_processed, duplicate_sets_found,
                           space_saved_mb, status, notes
                    FROM deduplication_sessions
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row['session_id'],
                        'created_at': row['created_at'],
                        'similarity_threshold': row['similarity_threshold'],
                        'total_images_processed': row['total_images_processed'],
                        'duplicate_sets_found': row['duplicate_sets_found'],
                        'space_saved_mb': row['space_saved_mb'],
                        'status': row['status'],
                        'notes': row['notes']
                    })
                
                return sessions
                
        except Exception as e:
            print(f"Error getting deduplication history: {e}")
            return []
    
    def get_processed_files(self, session_id: str = None, 
                           file_path: str = None) -> List[Dict]:
        """
        Get files processed in deduplication sessions.
        
        Args:
            session_id: Filter by specific session
            file_path: Filter by specific file path
            
        Returns:
            List of processed file records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM deduplication_files WHERE 1=1"
                params = []
                
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                
                if file_path:
                    query += " AND file_path = ?"
                    params.append(file_path)
                
                query += " ORDER BY processed_at DESC"
                
                cursor.execute(query, params)
                
                files = []
                for row in cursor.fetchall():
                    files.append({
                        'session_id': row['session_id'],
                        'file_path': row['file_path'],
                        'file_hash': row['file_hash'],
                        'file_size': row['file_size'],
                        'duplicate_group_id': row['duplicate_group_id'],
                        'is_keeper': row['is_keeper'],
                        'action_taken': row['action_taken'],
                        'moved_to_path': row['moved_to_path'],
                        'similarity_score': row['similarity_score'],
                        'processed_at': row['processed_at']
                    })
                
                return files
                
        except Exception as e:
            print(f"Error getting processed files: {e}")
            return []
    
    def has_been_deduplicated(self, file_path: str) -> bool:
        """
        Check if a file has been processed in any deduplication session.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file has been processed before, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM deduplication_files 
                    WHERE file_path = ? AND action_taken != 'skipped'
                """, (file_path,))
                
                result = cursor.fetchone()
                return result['count'] > 0 if result else False
                
        except Exception:
            return False