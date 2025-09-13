import { AwsClient } from "aws4fetch";

export interface Env {
  ASSETS: Fetcher;
  DOWNLOADS: R2Bucket;
  // R2 credentials for presigned URLs
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;
  R2_ACCOUNT_ID: string;
}

export const DOWNLOAD_FILES = {
  macos: "photo-grouper-macos.dmg",
  windows: "photo-grouper-windows.exe",
  linux: "photo-grouper-linux.AppImage",
} as const;

// Helper function to generate presigned URLs
async function generatePresignedUrl(
  env: Env,
  bucketName: string,
  objectKey: string,
  expiresIn: number = 3600
): Promise<string> {
  const client = new AwsClient({
    accessKeyId: env.R2_ACCESS_KEY_ID,
    secretAccessKey: env.R2_SECRET_ACCESS_KEY,
  });
  // https://979c4739d0e0ec53a3eb20186862ec6c.r2.cloudflarestorage.com/photo-grouper-downloads
  const r2Url = `https://${env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com/${bucketName}/${objectKey}`;

  const signedRequest = await client.sign(
    new Request(r2Url, { method: "GET" }),
    {
      aws: {
        signQuery: true,
        service: "s3",
        region: "auto",
      },
    }
  );

  return signedRequest.url;
}

// Helper function to check for compressed versions first and generate presigned URL
export async function getAvailableFile(
  bucket: R2Bucket,
  baseName: string,
  env: Env
): Promise<{
  fileName: string;
  downloadUrl: string;
  fileSize: number;
  expiresIn: number;
  isCompressed: boolean;
  compressionType?: string;
} | null> {
  // Try different compression formats in order of preference
  // Only use gzip for web compatibility
  const compressionFormats = [{ ext: ".gz", type: "gzip" }];

  for (const format of compressionFormats) {
    const compressedName = `${baseName}${format.ext}`;
    const compressedObject = await bucket.head(compressedName);
    if (compressedObject) {
      // Generate presigned URL using aws4fetch
      const presignedUrl = await generatePresignedUrl(
        env,
        "photo-grouper-downloads", // bucket name from wrangler.jsonc
        compressedName,
        3600
      );

      return {
        fileName: compressedName,
        downloadUrl: presignedUrl,
        fileSize: compressedObject.size,
        expiresIn: 3600, // 1 hour in seconds
        isCompressed: true,
        compressionType: format.type,
      };
    }
  }

  // Finally try uncompressed version
  const uncompressedObject = await bucket.head(baseName);
  if (uncompressedObject) {
    // Generate presigned URL using aws4fetch
    const presignedUrl = await generatePresignedUrl(
      env,
      "photo-grouper-downloads", // bucket name from wrangler.jsonc
      baseName,
      3600
    );

    return {
      fileName: baseName,
      downloadUrl: presignedUrl,
      fileSize: uncompressedObject.size,
      expiresIn: 3600, // 1 hour in seconds
      isCompressed: false,
    };
  }

  return null;
}

export const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};
