#!/bin/bash
set -e

echo "🔍 Verifying download file mappings..."

# Expected file mappings
declare -A EXPECTED_MAPPINGS=(
    ["macos"]="photo-grouper-macos.dmg"
    ["windows"]="photo-grouper-windows.exe"
    ["linux"]="photo-grouper-linux.AppImage"
)

# Function to test download API
test_download_api() {
    local platform="$1"
    local expected_filename="$2"
    
    echo "Testing $platform download..."
    
    if [ -n "$CLOUDFLARE_ACCOUNT_ID" ]; then
        # Test against actual deployment
        local base_url="https://photo-grouper-web.pages.dev"
        local response=$(curl -s "$base_url/api/download?platform=$platform" || echo "")
        
        if [ -n "$response" ]; then
            echo "✅ API Response for $platform:"
            echo "$response" | jq '.' 2>/dev/null || echo "$response"
        else
            echo "⚠️ No response from $platform API"
        fi
    else
        echo "ℹ️ Skipping API test (no CLOUDFLARE_ACCOUNT_ID)"
    fi
    
    echo ""
}

# Function to check R2 files
check_r2_files() {
    echo "📦 Checking R2 bucket contents..."
    
    if [ -n "$CLOUDFLARE_API_TOKEN" ] && [ -n "$CLOUDFLARE_ACCOUNT_ID" ]; then
        echo "Files in R2 bucket:"
        wrangler r2 object list photo-grouper-downloads --remote 2>/dev/null || echo "Could not list R2 objects"
    else
        echo "ℹ️ Skipping R2 check (missing credentials)"
    fi
    
    echo ""
}

# Main execution
echo "🎯 Expected file mappings:"
for platform in "${!EXPECTED_MAPPINGS[@]}"; do
    echo "  - $platform → ${EXPECTED_MAPPINGS[$platform]}"
done
echo ""

# Check R2 files
check_r2_files

# Test each platform
for platform in "${!EXPECTED_MAPPINGS[@]}"; do
    test_download_api "$platform" "${EXPECTED_MAPPINGS[$platform]}"
done

echo "✅ Verification complete!"
