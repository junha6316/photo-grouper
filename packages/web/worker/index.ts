interface Env {
  DOWNLOADS: R2Bucket;
}

const DOWNLOAD_FILES = {
  macos: "photo-grouper-macos.dmg",
  windows: "photo-grouper-windows.exe",
  linux: "photo-grouper-linux.AppImage",
} as const;

// Helper function to check for compressed versions first
async function getAvailableFile(
  bucket: R2Bucket,
  baseName: string
): Promise<{ fileName: string; isCompressed: boolean } | null> {
  // First try compressed version
  const compressedName = `${baseName}.gz`;
  const compressedObject = await bucket.head(compressedName);
  if (compressedObject) {
    return { fileName: compressedName, isCompressed: true };
  }

  // Then try uncompressed version
  const uncompressedObject = await bucket.head(baseName);
  if (uncompressedObject) {
    return { fileName: baseName, isCompressed: false };
  }

  return null;
}

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url);

    // CORS headers for all API requests
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    // Handle preflight requests
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    if (url.pathname === "/api/") {
      return Response.json(
        {
          name: "Photo Grouper Download API",
          version: "1.0.0",
        },
        { headers: corsHeaders }
      );
    }

    // Download endpoint
    if (url.pathname === "/api/download") {
      const platform = url.searchParams.get("platform");

      if (!platform || !(platform in DOWNLOAD_FILES)) {
        return Response.json(
          { error: "Invalid platform. Use: macos, windows, or linux" },
          { status: 400, headers: corsHeaders }
        );
      }

      const baseName = DOWNLOAD_FILES[platform as keyof typeof DOWNLOAD_FILES];

      try {
        // Check for available file (compressed or uncompressed)
        const fileInfo = await getAvailableFile(env.DOWNLOADS, baseName);
        if (!fileInfo) {
          return Response.json(
            { error: "Download file not found" },
            { status: 404, headers: corsHeaders }
          );
        }

        // Get file metadata
        const object = await env.DOWNLOADS.head(fileInfo.fileName);
        if (!object) {
          return Response.json(
            { error: "Download file not found" },
            { status: 404, headers: corsHeaders }
          );
        }

        // Generate download URL that points back to our worker
        const downloadUrl = `${url.origin}/api/download-file?platform=${platform}`;

        return Response.json(
          {
            platform,
            fileName: fileInfo.isCompressed ? baseName : fileInfo.fileName, // Return original name for user
            actualFileName: fileInfo.fileName, // Internal file name (may be compressed)
            downloadUrl,
            fileSize: object.size,
            isCompressed: fileInfo.isCompressed,
            expiresIn: 3600,
          },
          { headers: corsHeaders }
        );
      } catch (error) {
        console.error("Error generating download URL:", error);
        return Response.json(
          { error: "Failed to generate download URL" },
          { status: 500, headers: corsHeaders }
        );
      }
    }

    // Direct file download endpoint
    if (url.pathname === "/api/download-file") {
      const platform = url.searchParams.get("platform");

      if (!platform || !(platform in DOWNLOAD_FILES)) {
        return new Response("Invalid platform", { status: 400 });
      }

      const baseName = DOWNLOAD_FILES[platform as keyof typeof DOWNLOAD_FILES];

      try {
        // Check for available file (compressed or uncompressed)
        const fileInfo = await getAvailableFile(env.DOWNLOADS, baseName);
        if (!fileInfo) {
          return new Response("File not found", { status: 404 });
        }

        // Get file from R2
        const object = await env.DOWNLOADS.get(fileInfo.fileName);
        if (!object) {
          return new Response("File not found", { status: 404 });
        }

        // Set appropriate headers for file download
        const headers = new Headers();
        if (fileInfo.isCompressed) {
          headers.set("Content-Type", "application/gzip");
          headers.set("Content-Encoding", "gzip");
          headers.set(
            "Content-Disposition",
            `attachment; filename="${baseName}"` // Use original name
          );
        } else {
          headers.set("Content-Type", "application/octet-stream");
          headers.set(
            "Content-Disposition",
            `attachment; filename="${fileInfo.fileName}"`
          );
        }
        headers.set("Content-Length", object.size.toString());

        // Add CORS headers
        headers.set("Access-Control-Allow-Origin", "*");

        return new Response(object.body, { headers });
      } catch (error) {
        console.error("Error downloading file:", error);
        return new Response("Download failed", { status: 500 });
      }
    }

    return new Response(null, { status: 404 });
  },
} satisfies ExportedHandler<Env>;
