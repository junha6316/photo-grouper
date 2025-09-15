#!/bin/bash

set -e

# Check for required base icon
if [ ! -f "assets/app-icon.png" ]; then
    echo "ERROR: assets/app-icon.png not found. Please add your base PNG icon (1024x1024 with transparency recommended)." >&2
    exit 1
fi

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Generating macOS icons..."

    # Clean and create iconset directory
    rm -rf assets/AppIcon.iconset
    mkdir -p assets/AppIcon.iconset

    # Generate all required sizes for macOS
    sips -z 16 16     assets/app-icon.png --out assets/AppIcon.iconset/icon_16x16.png
    sips -z 32 32     assets/app-icon.png --out assets/AppIcon.iconset/icon_16x16@2x.png
    sips -z 32 32     assets/app-icon.png --out assets/AppIcon.iconset/icon_32x32.png
    sips -z 64 64     assets/app-icon.png --out assets/AppIcon.iconset/icon_32x32@2x.png
    sips -z 128 128   assets/app-icon.png --out assets/AppIcon.iconset/icon_128x128.png
    sips -z 256 256   assets/app-icon.png --out assets/AppIcon.iconset/icon_128x128@2x.png
    sips -z 256 256   assets/app-icon.png --out assets/AppIcon.iconset/icon_256x256.png
    sips -z 512 512   assets/app-icon.png --out assets/AppIcon.iconset/icon_256x256@2x.png
    sips -z 512 512   assets/app-icon.png --out assets/AppIcon.iconset/icon_512x512.png
    # 1024x1024 for @2x
    cp assets/app-icon.png assets/AppIcon.iconset/icon_512x512@2x.png

    # Create ICNS file
    iconutil -c icns assets/AppIcon.iconset -o assets/icon.icns
    echo "Created assets/icon.icns"
    ls -la assets/icon.icns

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OS" == "Windows_NT" ]]; then
    echo "Generating Windows icons..."

    # Check for ImageMagick
    if ! command -v magick &> /dev/null; then
        echo "ImageMagick not found. Installing via Chocolatey..."
        choco install imagemagick -y
    fi

    # Create Windows ICO with multiple sizes
    magick convert assets/app-icon.png -define icon:auto-resize=256,128,64,48,32,16 assets/icon.ico
    echo "Created assets/icon.ico"

else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "Icon generation completed successfully!"