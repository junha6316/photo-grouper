import type { R2Bucket } from "@cloudflare/workers-types";

export interface Env {
  DOWNLOADS: R2Bucket;
}

export const DOWNLOAD_FILES = {
  macos: "photo-grouper-macos.dmg",
  windows: "photo-grouper-windows.exe",
  linux: "photo-grouper-linux.AppImage",
} as const;

// Helper function to check for compressed versions first
export async function getAvailableFile(
  bucket: R2Bucket,
  baseName: string
): Promise<{
  fileName: string;
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
      return {
        fileName: compressedName,
        isCompressed: true,
        compressionType: format.type,
      };
    }
  }

  // Finally try uncompressed version
  const uncompressedObject = await bucket.head(baseName);
  if (uncompressedObject) {
    return { fileName: baseName, isCompressed: false };
  }

  return null;
}

export const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};
