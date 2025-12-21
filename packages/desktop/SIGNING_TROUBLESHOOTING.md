# Code Signing and Notarization Troubleshooting

## Error: "A required agreement is missing or has expired"

### Symptoms
```
Error: HTTP status code: 403. A required agreement is missing or has expired.
This request requires an in-effect agreement that has not been signed or has expired.
```

### Root Cause
Apple periodically updates their Developer Program License Agreement. When this happens, you must review and accept the new terms before you can notarize apps.

### Solution

#### Step 1: Accept Apple Developer Agreement

1. **Visit Apple Developer Portal**
   - Go to https://developer.apple.com/account
   - Log in with your Apple ID (the one associated with your Developer account)

2. **Check for Pending Agreements**
   - Look for a red banner or notification at the top of the page
   - If there's a pending agreement, you'll see a message like:
     > "Your membership requires you to review and accept the updated agreements"

3. **Review and Accept**
   - Click "Review Agreement" or "Accept"
   - Read the updated terms
   - Check the box to agree
   - Click "Submit" or "Accept"

4. **Verify Account Status**
   - After accepting, your account status should show as "Active"
   - No warnings or pending actions should remain

#### Step 2: Verify Team ID and Certificates

```bash
# Check your Team ID
# Visit: https://developer.apple.com/account -> Membership Details
# Your Team ID should match the one in your environment variables

# Verify code signing certificates
security find-identity -v -p codesigning

# You should see:
# - "Developer ID Application: Your Name (TEAM_ID)" - for distribution
# - "Apple Development: Your Name" - for development (optional)
```

#### Step 3: Test Notarization

After accepting the agreement, test the notarization process:

```bash
cd packages/desktop

# Set environment variables
export DEVELOPER_ID_NAME="Developer ID Application: Your Name (TEAM_ID)"
export APPLE_ID="your-apple-id@example.com"
export APPLE_ID_PASSWORD="xxxx-xxxx-xxxx-xxxx"  # App-specific password
export APPLE_TEAM_ID="YOUR_TEAM_ID"

# Run the sign and notarize script
./scripts/sign-and-notarize.sh
```

### Alternative: Use Xcode's Notarization Tool

If the command-line tool continues to fail, you can use Xcode:

```bash
# 1. Sign the app
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: Your Name (TEAM_ID)" \
  --options runtime \
  "dist/Photo Grouper.app"

# 2. Create DMG
create-dmg \
  --volname "Photo Grouper" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "Photo Grouper.app" 200 190 \
  --hide-extension "Photo Grouper.app" \
  --app-drop-link 600 185 \
  "photo-grouper-macos.dmg" \
  "dist/"

# 3. Notarize via Xcode
# Open Xcode -> Window -> Organizer -> Distribute App
# Select "Developer ID" and follow the wizard
```

### Common Issues

#### Issue: "Invalid Apple ID or Password"
**Solution**: Make sure you're using an **app-specific password**, not your regular Apple ID password.
- Generate at: https://appleid.apple.com (Sign In → Security → App-Specific Passwords)

#### Issue: "Team ID not found"
**Solution**: Verify your Team ID matches:
```bash
# Get Team ID from certificate
security find-identity -v -p codesigning

# It's the string in parentheses, e.g., (HN4J56P9K4)
```

#### Issue: "Certificate has expired"
**Solution**: Renew your Developer ID certificate:
1. Go to https://developer.apple.com/account/resources/certificates
2. Find "Developer ID Application" certificate
3. If expired, revoke and create a new one
4. Download and install in Keychain Access

### Verify Everything is Working

After fixing the agreement issue, verify:

```bash
# 1. Check account status
# Visit: https://developer.apple.com/account
# Status should be "Active" with no warnings

# 2. Test notarization with a simple file
echo "test" > test.txt
zip test.zip test.txt

xcrun notarytool submit test.zip \
  --apple-id "$APPLE_ID" \
  --password "$APPLE_ID_PASSWORD" \
  --team-id "$APPLE_TEAM_ID" \
  --wait

# 3. If successful, clean up and proceed with app notarization
rm test.txt test.zip
```

### For CI/CD (GitHub Actions)

The GitHub Actions workflow should work automatically once you've:
1. Accepted the Developer Agreement
2. Verified the following secrets are set correctly:
   - `MACOS_CERTIFICATE` (base64 of .p12 file)
   - `MACOS_CERTIFICATE_PWD` (.p12 password)
   - `APPLE_ID` (your Apple ID email)
   - `APPLE_ID_PASSWORD` (app-specific password)
   - `APPLE_TEAM_ID` (your team ID)
   - `DEVELOPER_ID_NAME` (full certificate name)

### Need More Help?

- Apple Developer Support: https://developer.apple.com/support/
- Notarization Guide: https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution
- Code Signing Guide: https://developer.apple.com/support/code-signing/
