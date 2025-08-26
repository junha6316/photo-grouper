# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python desktop application for grouping similar photos using ML embeddings and cosine similarity. Features real-time regrouping via threshold slider, async image loading, and comprehensive selection/export capabilities.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Development Commands

```bash
# Clean Python cache files
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Clear embedding cache (SQLite database)
rm -rf ~/.photo_grouper/embeddings.db

# Monitor memory usage (requires memory_profiler)
python -m memory_profiler app.py
```

## Architecture

### Core Processing (`core/`)
- `scanner.py` - Recursive image discovery, supports JPG/PNG/HEIC/HEIF formats
- `embedder.py` - VGG16 feature extraction (512D) → PCA reduction → L2 normalization
- `grouper.py` - NetworkX connected components for grouping, spectral ordering for cluster arrangement
- `deduplicator.py` - File hash-based duplicate detection
- `select_session.py` - Session management for grouped selections

### Infrastructure (`infra/`)
- `cache_db.py` - SQLite embedding cache with (path, mtime, hash) validation

### User Interface (`ui/`)
- `main_window.py` - QMainWindow with threaded processing, manages global selection state
- `preview_panel.py` - Thumbnail grid with lazy loading  
- `cluster_dialog.py` - Detail view with flow layout, selection, zoom controls
- `selected_view_dialog.py` - Grid view of selected images
- `async_image_loader.py` - Priority queue-based background loading
- `session_launcher.py` - Multi-session management UI
- `deduplication_dialog.py` - Duplicate management interface

### UI Components (`ui/components/`)
- `zoomable_image_label.py` - Pan/zoom image viewer widget
- `image_widgets.py` - Reusable image display widgets
- `layouts.py` - Custom layout managers
- `panels.py` - Specialized panel widgets

## Key Implementation Details

### Embedding Pipeline
1. Load image → Resize to 224×224 → VGG16 features (512D from GAP)
2. Fit PCA on sample (up to 1000 images) → Transform all embeddings
3. L2 normalize → Store in SQLite cache

### Grouping Algorithm (NetworkX-based)
```python
# core/grouper.py:_group_by_direct_similarity()
1. Compute full cosine similarity matrix
2. Create graph with edges where similarity >= threshold  
3. Find connected components (includes singletons)
4. Sort clusters by inter-cluster similarity
```

### Cache Strategy
- **SQLite**: Persistent cache at `~/.photo_grouper/embeddings.db`
- **Memory**: In-process dict cache for current session
- **Validation**: File hash from `(path, size, mtime)`

### Threading Model
- **Main Thread**: Qt event loop, UI updates
- **ProcessingThread**: Scanning → Embedding → Grouping pipeline
- **AsyncImageLoader**: Singleton worker with priority queue

## Performance Optimizations

- Tiled similarity computation for memory efficiency (tile_size=1000)
- FAISS acceleration available for datasets >10k images
- Lazy thumbnail loading with viewport prioritization
- PCA dimensionality reduction (512→512, preserves variance)

## File Formats Support

```python
SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
    '.webp', '.heic', '.heif'
}
```

## UI State Management

- Global selection tracked in `MainWindow.global_selected_images`
- Cluster dialogs sync selection state via signals
- Preview panel updates on threshold changes

## Database Schema

```sql
CREATE TABLE embeddings (
    file_path TEXT,
    file_hash TEXT,
    file_size INTEGER,
    file_mtime REAL,
    model_name TEXT,  -- 'vgg16_raw' or 'vgg16_pca'
    embedding_data BLOB,
    UNIQUE(file_path, file_hash, model_name)
)
```

## Configuration Parameters

- **Similarity threshold**: 0.50-0.99 (default 0.85)
- **Min group size**: 1 (singles always grouped together)
- **Thumbnail sizes**: 90px (main), 150px (detail)
- **PCA components**: 512 (matches VGG16 output)
- **Tile size**: 1000 (for similarity computation)