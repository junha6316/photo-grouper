#!/usr/bin/env python3
"""
Cache management utility for photo grouper.
Usage: python scripts/manage_cache.py [command] [options]
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.cache_db import EmbeddingCache
from infra.thumbnail_cache import ThumbnailCache


def format_size(size_bytes):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def show_stats(args):
    """Show cache statistics."""
    print("\n=== Photo Grouper Cache Statistics ===\n")
    
    # Embedding cache stats
    embed_cache = EmbeddingCache()
    embed_stats = embed_cache.get_cache_stats()
    
    print("Embedding Cache:")
    print(f"  Total embeddings: {embed_stats['total_embeddings']}")
    for model, count in embed_stats.get('by_model', {}).items():
        print(f"    {model}: {count} embeddings")
    
    # Thumbnail cache stats
    thumb_cache = ThumbnailCache()
    thumb_stats = thumb_cache.get_cache_stats()
    
    print("\nThumbnail Cache:")
    print(f"  Total thumbnails: {thumb_stats['total_thumbnails']}")
    print(f"  Total size: {thumb_stats['total_size_mb']:.2f} MB")
    
    if thumb_stats['by_size']:
        print("  By size:")
        for size, info in thumb_stats['by_size'].items():
            print(f"    {size}px: {info['count']} thumbnails ({info['size_mb']:.2f} MB)")
    
    # Database file sizes
    cache_dir = Path.home() / ".photo_grouper"
    if cache_dir.exists():
        print("\nCache Files:")
        for db_file in cache_dir.glob("*.db"):
            size = db_file.stat().st_size
            print(f"  {db_file.name}: {format_size(size)}")


def clear_cache(args):
    """Clear cache based on type."""
    if args.type == 'all' or args.type == 'embeddings':
        embed_cache = EmbeddingCache()
        count = embed_cache.clear_cache(args.model)
        print(f"Cleared {count} embedding entries")
    
    if args.type == 'all' or args.type == 'thumbnails':
        thumb_cache = ThumbnailCache()
        size_arg = int(args.size) if args.size else None
        count = thumb_cache.clear_cache(size_arg)
        print(f"Cleared {count} thumbnail entries")
    
    print("Cache cleared successfully")


def cleanup_cache(args):
    """Remove invalid cache entries."""
    print("Cleaning up invalid cache entries...")
    
    embed_cache = EmbeddingCache()
    embed_removed = embed_cache.cleanup_invalid_entries()
    print(f"Removed {embed_removed} invalid embedding entries")
    
    thumb_cache = ThumbnailCache()
    thumb_removed = thumb_cache.cleanup_invalid_entries()
    print(f"Removed {thumb_removed} invalid thumbnail entries")
    
    print("Cleanup completed")


def precache_thumbnails(args):
    """Pre-generate thumbnails for a directory."""
    from PIL import Image
    
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return
    
    sizes = [int(s) for s in args.sizes.split(',')]
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'}
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to process")
    print(f"Generating thumbnails for sizes: {sizes}")
    
    thumb_cache = ThumbnailCache()
    processed = 0
    skipped = 0
    errors = 0
    
    for i, image_path in enumerate(image_files, 1):
        try:
            # Check each size
            for size in sizes:
                # Check if already cached
                if thumb_cache.get_thumbnail(str(image_path), size):
                    skipped += 1
                else:
                    # Generate and cache
                    thumb_cache.create_and_cache_thumbnail(str(image_path), size)
                    processed += 1
            
            # Progress indicator
            if i % 10 == 0:
                print(f"Progress: {i}/{len(image_files)} images...", end='\r')
        
        except Exception as e:
            errors += 1
            if args.verbose:
                print(f"\nError processing {image_path}: {e}")
    
    print(f"\nCompleted: {processed} thumbnails generated, {skipped} already cached, {errors} errors")


def main():
    parser = argparse.ArgumentParser(description='Manage photo grouper cache')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache')
    clear_parser.add_argument('--type', choices=['all', 'embeddings', 'thumbnails'],
                             default='all', help='Type of cache to clear')
    clear_parser.add_argument('--model', help='Specific model name for embeddings')
    clear_parser.add_argument('--size', help='Specific thumbnail size to clear')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove invalid entries')
    
    # Precache command
    precache_parser = subparsers.add_parser('precache', help='Pre-generate thumbnails')
    precache_parser.add_argument('directory', help='Directory to process')
    precache_parser.add_argument('--sizes', default='150,256',
                                 help='Comma-separated thumbnail sizes (default: 150,256)')
    precache_parser.add_argument('--verbose', action='store_true',
                                 help='Show detailed error messages')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    commands = {
        'stats': show_stats,
        'clear': clear_cache,
        'cleanup': cleanup_cache,
        'precache': precache_thumbnails
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()