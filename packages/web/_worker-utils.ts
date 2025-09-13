// R2 Bucket type definition
interface R2Object {
  key: string;
  url?: string;
  size: number;
}

interface R2Bucket {
  head(key: string): Promise<{ size: number; etag: string } | null>;
  get(key: string): Promise<R2Object | null>;
}

export interface Env {
  ASSETS: Fetcher;
  DOWNLOADS: R2Bucket;
}

export const DOWNLOAD_FILES = {
  macos: "photo-grouper-macos.dmg",
  windows: "photo-grouper-windows.exe",
  linux: "photo-grouper-linux.AppImage",
} as const;

// Helper function to check for compressed versions first and generate presigned URL
export async function getAvailableFile(
  bucket: R2Bucket,
  baseName: string
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
      // Generate presigned URL - R2 automatically creates a presigned URL when you get the object
      const r2Object = await bucket.get(compressedName);

      if (r2Object) {
        return {
          fileName: compressedName,
          downloadUrl: r2Object.url || "", // This is the presigned URL
          fileSize: compressedObject.size,
          expiresIn: 3600, // 1 hour in seconds
          isCompressed: true,
          compressionType: format.type,
        };
      }
    }
  }

  // Finally try uncompressed version
  const uncompressedObject = await bucket.head(baseName);
  if (uncompressedObject) {
    // Generate presigned URL - R2 automatically creates a presigned URL when you get the object
    const r2Object = await bucket.get(baseName);

    if (r2Object) {
      return {
        fileName: baseName,
        downloadUrl: r2Object.url || "", // This is the presigned URL
        fileSize: uncompressedObject.size,
        expiresIn: 3600, // 1 hour in seconds
        isCompressed: false,
      };
    }
  }

  return null;
}

export const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};
