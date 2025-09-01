# Photo Grouper

A monorepo containing intelligent photo organization tools using machine learning embeddings and cosine similarity. Currently includes a desktop application built with Python and PySide6, with plans for web-based distribution.

![Photo Grouper Screenshot](assets/image.png)

## Features

- **Smart Photo Grouping**: Automatically groups similar photos using deep learning embeddings
- **Real-time Threshold Adjustment**: Dynamically regroup photos by adjusting similarity threshold with a slider
- **Duplicate Detection**: Identifies and manages duplicate images based on file hashes
- **Async Image Loading**: Smooth UI experience with background image loading and priority queuing
- **Multiple Sessions**: Manage different grouping sessions for various photo collections
- **Export Functionality**: Export selected groups to organized folders
- **Format Support**: Works with JPG, PNG, HEIC, HEIF, BMP, TIFF, and WebP formats

## Installation

### Prerequisites

- [mise](https://mise.jdx.dev/) - Runtime version manager
- Python 3.13+ (managed by mise)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/junha6316/photo-grouper.git
cd photo-grouper
```

2. Install mise (if not already installed):

```bash
# macOS
brew install mise

# Linux/Windows - see https://mise.jdx.dev/getting-started.html
```

3. Install dependencies:

```bash
mise install
mise run install
```

## Usage

### Basic Usage

Run the desktop application:

```bash
mise run run-desktop
```

Or from the desktop package directory:

```bash
cd packages/desktop
mise run run
```

### Application Workflow

1. Click "Select Folder" to choose a directory containing images
2. Wait for the application to scan and process images
3. Adjust the similarity threshold slider (0.50-0.99) to change grouping sensitivity
4. Click on any group to view detailed images
5. Select images and export them to organized folders

### Features Guide

#### Similarity Threshold

- **Higher values (0.90-0.99)**: Groups only very similar photos
- **Medium values (0.80-0.89)**: Balanced grouping for most use cases
- **Lower values (0.50-0.79)**: Groups moderately similar photos

#### Selection Management

- Click images to select/deselect
- Use "Select All" / "Deselect All" buttons in group views
- View all selected images via "View Selected" button
- Export selected images to a target folder

#### Duplicate Management

- Access via "Find Duplicates" button
- Review and manage exact duplicate files
- Keep one copy and remove others safely

## Technical Details

### Architecture

The desktop application uses a modular architecture within `packages/desktop/`:

- **Core Processing** (`core/`): Image scanning, embedding generation, similarity grouping
- **Infrastructure** (`infra/`): SQLite caching, persistence layer
- **User Interface** (`ui/`): PySide6-based GUI with responsive components
- **UI Components** (`ui/components/`): Reusable widgets and layouts

### ML Pipeline

1. **Feature Extraction**: VGG16 convolutional neural network (pre-trained on ImageNet)
2. **Dimensionality Reduction**: PCA to optimize computation while preserving variance
3. **Similarity Computation**: Cosine similarity between L2-normalized embeddings
4. **Graph-based Grouping**: NetworkX connected components for cluster formation

### Performance

- **Caching**: SQLite database stores computed embeddings to avoid reprocessing
- **Tiled Processing**: Memory-efficient similarity computation for large datasets
- **Lazy Loading**: Images loaded on-demand with viewport prioritization
- **Threading**: Background processing keeps UI responsive

## Development

### Monorepo Structure

```
photo-grouper/
├── .mise.toml             # Development environment configuration
├── assets/                # Shared assets
│   └── image.png         # Project screenshots
├── packages/
│   └── desktop/          # Desktop application
│       ├── app.py        # Application entry point
│       ├── core/         # Core processing logic
│       │   ├── scanner.py    # Image discovery
│       │   ├── embedder.py   # ML feature extraction
│       │   ├── grouper.py    # Similarity grouping
│       │   └── deduplicator.py # Duplicate detection
│       ├── infra/        # Infrastructure layer
│       │   └── cache_db.py   # Embedding cache
│       ├── ui/           # User interface
│       │   ├── components/   # Reusable UI components
│       │   └── views/        # Application views
│       ├── windows/      # Main window implementation
│       ├── pyproject.toml    # Python project configuration
│       └── requirements.txt  # Python dependencies
├── README.md             # This file
└── LICENSE              # MIT License
```

### Development Commands

All development tasks are managed through mise tasks:

```bash
# Development workflow
mise run install           # Install dependencies
mise run run-desktop       # Run desktop application
mise run clean            # Clean cache files

# Code quality
mise run format           # Format code (black, isort)
mise run lint             # Run linter (ruff)
mise run type-check       # Type checking (mypy)
mise run check            # Run all checks

# Testing & building
mise run test             # Run tests
mise tasks                # List all available tasks

# Traditional commands (if needed)
cd packages/desktop
uv run python app.py      # Direct app execution
```

### Utility Commands

```bash
# Clear embedding cache
rm -rf ~/.photo_grouper/embeddings.db

# Run with memory profiling (from packages/desktop/)
uv run python -m memory_profiler app.py
```

## Configuration

The application stores configuration and cache in:

- **Cache**: `~/.photo_grouper/embeddings.db`
- **Settings**: Platform-specific user config directory

## Requirements

### System Requirements

- **Runtime Manager**: [mise](https://mise.jdx.dev/) for environment management
- **Python**: 3.13+ (automatically installed by mise)
- **Package Manager**: uv (used within the desktop package)

### Key Dependencies (Desktop Package)

- **PySide6** - Modern Qt6 GUI framework
- **NumPy** - Numerical computations
- **Pillow** - Image processing with HEIC/HEIF support
- **scikit-learn** - PCA and preprocessing
- **PyTorch & torchvision** - Neural network models (VGG16, ResNet18, MobileNet)
- **NetworkX** - Graph algorithms for grouping
- **FAISS** - Fast similarity search (optional acceleration)

## Future Plans

This monorepo is designed for expansion:

### 🌐 Web Package (`packages/web/`)

- **Hono-based API** - Fast web framework for download/distribution
- **Progressive Web App** - Browser-based photo grouping
- **Cloud Processing** - Server-side ML inference

### 🔧 Shared Utilities (`shared/`)

- **Common algorithms** - Shared ML models and utilities
- **Asset management** - Unified branding and documentation
- **Configuration** - Centralized settings management

### 🚀 CI/CD Pipeline

- **Automated testing** - Desktop and web package testing
- **Multi-platform builds** - Windows, macOS, Linux distributions
- **Web deployment** - Automated deployment to cloud platforms

## License

MIT License - see the [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! This monorepo supports multiple development workflows:

### Desktop Package

```bash
cd packages/desktop
mise run install-dev
mise run format
mise run check
```

### Future Packages

As we add `packages/web/` and other packages, each will have its own development workflow while sharing common tools through the root `.mise.toml`.

Please feel free to submit a Pull Request!

## Support

For issues and questions:

- **Desktop Application**: Use GitHub issues with the `desktop` label
- **General Questions**: Use GitHub Discussions
- **Feature Requests**: Use GitHub issues with the `enhancement` label
