#!/bin/bash

# macOS Code Signing and Notarization Script
# This script signs and notarizes the Photo Grouper app bundle
# Usage: ./scripts/sign-and-notarize.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check required environment variables
check_env_vars() {
    local missing_vars=()

    if [ -z "$DEVELOPER_ID_NAME" ]; then
        missing_vars+=("DEVELOPER_ID_NAME")
    fi
    if [ -z "$APPLE_ID" ]; then
        missing_vars+=("APPLE_ID")
    fi
    if [ -z "$APPLE_ID_PASSWORD" ]; then
        missing_vars+=("APPLE_ID_PASSWORD")
    fi
    if [ -z "$APPLE_TEAM_ID" ]; then
        missing_vars+=("APPLE_TEAM_ID")
    fi

    if [ ${#missing_vars[@]} -ne 0 ]; then
        echo -e "${RED}❌ Missing required environment variables:${NC}"
        for var in "${missing_vars[@]}"; do
            echo "   - $var"
        done
        echo ""
        echo -e "${YELLOW}Please set the following environment variables:${NC}"
        echo ""
        echo "export DEVELOPER_ID_NAME=\"Developer ID Application: Your Name (TEAM_ID)\""
        echo "export APPLE_ID=\"your-apple-id@example.com\""
        echo "export APPLE_ID_PASSWORD=\"app-specific-password\""
        echo "export APPLE_TEAM_ID=\"YOUR_TEAM_ID\""
        echo ""
        echo -e "${YELLOW}How to get these values:${NC}"
        echo "1. DEVELOPER_ID_NAME: Run 'security find-identity -v -p codesigning' to find your Developer ID"
        echo "2. APPLE_ID: Your Apple ID email"
        echo "3. APPLE_ID_PASSWORD: App-specific password from https://appleid.apple.com"
        echo "4. APPLE_TEAM_ID: Your Apple Developer Team ID from developer.apple.com"
        exit 1
    fi
}

# Check if app bundle exists
check_app_bundle() {
    if [ ! -d "dist/Photo Grouper.app" ]; then
        echo -e "${RED}❌ App bundle not found at dist/Photo Grouper.app${NC}"
        echo "Please build the app first using: uv run pyinstaller ./photo-grouper.spec"
        exit 1
    fi
}

# Sign the application
sign_app() {
    echo -e "${GREEN}🔐 Signing Photo Grouper.app...${NC}"

    codesign --deep --force --verify --verbose \
        --sign "$DEVELOPER_ID_NAME" \
        --options runtime \
        "dist/Photo Grouper.app"

    echo -e "${GREEN}✅ Verifying signature...${NC}"
    codesign --verify --verbose "dist/Photo Grouper.app"

    echo -e "${GREEN}✅ Application signed successfully${NC}"
}

# Notarize the application
notarize_app() {
    echo -e "${GREEN}📦 Creating zip for notarization...${NC}"
    ditto -c -k --keepParent "dist/Photo Grouper.app" "Photo Grouper.zip"

    echo -e "${GREEN}🚀 Submitting for notarization...${NC}"
    echo "This may take several minutes..."

    xcrun notarytool submit "Photo Grouper.zip" \
        --apple-id "$APPLE_ID" \
        --password "$APPLE_ID_PASSWORD" \
        --team-id "$APPLE_TEAM_ID" \
        --wait

    echo -e "${GREEN}📌 Stapling notarization ticket...${NC}"
    xcrun stapler staple "dist/Photo Grouper.app"

    # Clean up zip file
    rm "Photo Grouper.zip"

    echo -e "${GREEN}✅ Application notarized successfully${NC}"
}

# Verify the final result
verify_app() {
    echo -e "${GREEN}🔍 Verifying final app bundle...${NC}"

    echo "Code signature:"
    codesign --display --verbose=4 "dist/Photo Grouper.app"

    echo ""
    echo "Notarization status:"
    xcrun stapler validate "dist/Photo Grouper.app"

    echo ""
    echo "Gatekeeper assessment:"
    spctl -a -v "dist/Photo Grouper.app"
}

# Main execution
main() {
    echo -e "${GREEN}=== Photo Grouper Code Signing and Notarization ===${NC}"
    echo ""

    check_env_vars
    check_app_bundle

    echo ""
    echo -e "${YELLOW}📝 Configuration:${NC}"
    echo "  Developer ID: $DEVELOPER_ID_NAME"
    echo "  Apple ID: $APPLE_ID"
    echo "  Team ID: $APPLE_TEAM_ID"
    echo ""

    read -p "Continue with signing and notarization? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi

    sign_app
    echo ""
    notarize_app
    echo ""
    verify_app

    echo ""
    echo -e "${GREEN}🎉 All done! Your app is now signed and notarized.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Create DMG: brew install create-dmg && create-dmg --help"
    echo "2. Or distribute the .app bundle directly"
}

main
