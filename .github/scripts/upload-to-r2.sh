#!/bin/bash
set -e

# Upload file to Cloudflare R2 with multipart support
upload_to_r2() {
    local file_path="$1"
    local remote_key="$2"
    
    if [ ! -f "$file_path" ]; then
        echo "File not found: $file_path"
        return 1
    fi
    
    echo "Uploading $file_path..."
    file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null)
    echo "File size: $file_size bytes"
    
    # Determine content-type
    if [[ "$file_path" == *.gz ]]; then
        content_type="application/gzip"
        content_encoding="--content-encoding gzip"
    elif [[ "$file_path" == *.xz ]]; then
        content_type="application/x-xz"
        content_encoding="--content-encoding xz"
    else
        content_type="application/octet-stream"
        content_encoding=""
    fi
    
    # Try S3-compatible API with multipart support first
    if [ -n "$R2_ACCESS_KEY_ID" ] && [ -n "$R2_SECRET_ACCESS_KEY" ]; then
        echo "Using S3-compatible API with multipart support"
        
        # Configure AWS CLI
        aws configure set aws_access_key_id "$R2_ACCESS_KEY_ID"
        aws configure set aws_secret_access_key "$R2_SECRET_ACCESS_KEY"
        aws configure set region auto
        aws configure set s3.multipart_threshold 64MB
        aws configure set s3.multipart_chunksize 16MB
        
        R2_ENDPOINT="https://$CLOUDFLARE_ACCOUNT_ID.r2.cloudflarestorage.com"
        
        if aws s3 cp "$file_path" "s3://photo-grouper-downloads/$remote_key" \
           --endpoint-url "$R2_ENDPOINT" \
           --content-type "$content_type" \
           $content_encoding \
           --no-progress; then
            echo "✅ Successfully uploaded via S3 API ($file_size bytes)"
            return 0
        else
            echo "⚠️ S3 API upload failed, trying Wrangler..."
        fi
    fi
    
    # Fallback to Wrangler
    echo "Using Wrangler for upload..."
    if wrangler r2 object put "$remote_key" \
       --file "$file_path" \
       --content-type "$content_type" \
       --remote; then
        echo "✅ Successfully uploaded via Wrangler ($file_size bytes)"
    else
        echo "❌ All upload methods failed for $file_path"
        return 1
    fi
}

# Main execution
upload_to_r2 "$1" "$2"
