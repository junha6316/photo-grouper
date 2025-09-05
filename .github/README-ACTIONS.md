# GitHub Actions Setup Guide

This repository uses GitHub Actions to automatically build and deploy the Photo Grouper desktop application when changes are made to the `packages/desktop/` directory.

## 🚀 What the Workflow Does

### Triggers

- **Push to `main` or `develop`**: Builds and deploys to Cloudflare R2
- **Push to `packages/desktop/`**: Builds for all platforms
- **Pull Request**: Builds only (no deployment)

### Build Process

1. **Multi-platform builds**: macOS, Windows, Linux
2. **PyInstaller packaging**: Creates standalone executables
3. **Platform-specific packaging**:
   - **macOS**: Creates `.dmg` installer
   - **Windows**: Creates `.exe` executable
   - **Linux**: Creates `.AppImage` portable app
4. **Automatic upload**: Deploys to Cloudflare R2 storage

## 📋 Setup Requirements

### 1. Repository Secrets

Add these secrets in GitHub Settings → Secrets and variables → Actions:

```
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
```

### 2. Cloudflare R2 Bucket

```bash
# Create the bucket (run once)
wrangler r2 bucket create photo-grouper-downloads
```

### 3. Icon Assets (Optional)

Create these icon files in `assets/` directory:

- `assets/icon.icns` (macOS)
- `assets/icon.ico` (Windows)
- `assets/icon.png` (Linux)

If icons don't exist, the workflow will create placeholder files.

## 📁 Output Files

After successful build, these files are uploaded to R2:

- `photo-grouper-macos.dmg` (or `photo-grouper-macos.dmg.gz` if compressed)
- `photo-grouper-windows.exe` (or `photo-grouper-windows.exe.gz` if compressed)
- `photo-grouper-linux.AppImage` (or `photo-grouper-linux.AppImage.gz` if compressed)

**File Size Handling**: Files larger than 300MB are automatically compressed with gzip before upload to comply with Wrangler's file size limits. The web download service automatically detects and serves both compressed and uncompressed files.

## 🔧 Customization

### Modifying Build Process

Edit `.github/workflows/build-desktop.yml` to:

- Change build commands
- Add/remove platforms
- Modify packaging options
- Add code signing (for distribution)

### PyInstaller Configuration

Edit `packages/desktop/photo-grouper.spec` to:

- Include/exclude modules
- Add hidden imports
- Modify app metadata
- Change icon paths

## 🐛 Troubleshooting

### Build Failures

1. Check the Actions tab for detailed logs
2. Verify all dependencies are in `pyproject.toml`
3. Ensure PyInstaller can find all modules

### File Size Issues

1. **"Wrangler only supports uploading files up to 300 MiB"**: Files are automatically compressed if they exceed 300MB
2. Check build logs to see if compression was applied
3. Large PyTorch/ML dependencies can cause size issues - consider optimizing excludes in `photo-grouper.spec`

### Missing Files

1. Verify the R2 bucket exists
2. Check Cloudflare API token permissions
3. Ensure secrets are set correctly
4. Check if files were compressed (look for `.gz` extensions in logs)

### Icon Issues

1. Icons are optional - builds will work without them
2. Use proper formats: `.icns` (macOS), `.ico` (Windows), `.png` (Linux)
3. Recommended size: 256x256 pixels

## 📈 Monitoring

### Build Status

- Check the "Actions" tab in your repository
- Green checkmark = successful build and deployment
- Red X = build failed (check logs)

### File Verification

```bash
# List files in R2 bucket
wrangler r2 object list photo-grouper-downloads

# Download and test files
wrangler r2 object get photo-grouper-downloads/photo-grouper-macos.dmg --file test.dmg
```

## 🔄 Workflow Files

- **`.github/workflows/build-desktop.yml`**: Main workflow
- **`.github/SECRETS.md`**: Detailed secrets setup
- **`packages/desktop/photo-grouper.spec`**: PyInstaller configuration

## 🚀 Next Steps

1. Set up the required secrets
2. Create the R2 bucket
3. Make a test change to `packages/desktop/`
4. Push to `main` branch
5. Watch the magic happen in the Actions tab! ✨
