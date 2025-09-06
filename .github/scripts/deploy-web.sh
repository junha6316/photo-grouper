#!/bin/bash
set -e

echo "🚀 Deploying web application to Cloudflare Pages..."

# Check if we're in the web package directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Make sure to run this from packages/web directory."
    exit 1
fi

# Check if required environment variables are set
if [ -z "$CLOUDFLARE_API_TOKEN" ] || [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
    echo "❌ Error: CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID must be set"
    exit 1
fi

# Get deployment environment
ENVIRONMENT="production"
if [ "$GITHUB_REF" != "refs/heads/main" ]; then
    ENVIRONMENT="preview"
fi

echo "📦 Environment: $ENVIRONMENT"
echo "🌿 Branch: ${GITHUB_REF_NAME:-$(git branch --show-current)}"

# Build the application
echo "🏗️ Building application..."
pnpm run build

# Deploy using Wrangler
echo "🚀 Deploying to Cloudflare Pages..."
if [ "$ENVIRONMENT" = "production" ]; then
    pnpm run deploy
else
    # For preview deployments, we might want different settings
    pnpm run deploy
fi

echo "✅ Deployment completed successfully!"

# Get deployment info
if command -v wrangler >/dev/null 2>&1; then
    echo "📊 Deployment info:"
    wrangler pages deployment list --project-name photo-grouper-web --limit 1 || true
fi
