#!/bin/bash
set -e

# Compress build artifacts with best available method
compress_artifact() {
    local artifact="$1"
    
    if [ ! -f "$artifact" ]; then
        echo "Warning: $artifact not found"
        return 1
    fi
    
    echo "Original file size: $(stat -f%z "$artifact" 2>/dev/null || stat -c%s "$artifact" 2>/dev/null) bytes"
    echo "Testing compression methods..."
    
    # Test gzip compression
    cp "$artifact" "$artifact.temp1"
    gzip -9 "$artifact.temp1"
    gzip_size=$(stat -f%z "$artifact.temp1.gz" 2>/dev/null || stat -c%s "$artifact.temp1.gz" 2>/dev/null)
    echo "gzip -9 size: $gzip_size bytes"
    
    # Use only gzip compression for web compatibility
    echo "Using gzip compression (web-compatible)"
    mv "$artifact.temp1.gz" "$artifact.gz"
    echo "COMPRESSION_EXT=.gz" >> $GITHUB_ENV
    final_size=$gzip_size
    
    # Calculate compression ratio
    original_size=$(stat -f%z "$artifact" 2>/dev/null || stat -c%s "$artifact" 2>/dev/null)
    ratio=$(echo "scale=2; ($original_size - $final_size) * 100 / $original_size" | bc -l 2>/dev/null || echo "N/A")
    echo "Compression ratio: ${ratio}%"
}

# Main execution
cd packages/desktop
compress_artifact "$1"
