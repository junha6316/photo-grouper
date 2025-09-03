interface Env {
  DOWNLOADS: R2Bucket;
}

const DOWNLOAD_FILES = {
  macos: "photo-grouper-macos.dmg",
  windows: "photo-grouper-windows.exe",
  linux: "photo-grouper-linux.AppImage",
} as const;

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

      const fileName = DOWNLOAD_FILES[platform as keyof typeof DOWNLOAD_FILES];

      try {
        // Check if file exists in R2
        const object = await env.DOWNLOADS.head(fileName);
        if (!object) {
          return Response.json(
            { error: "Download file not found" },
            { status: 404, headers: corsHeaders }
          );
        }

        // Generate presigned URL (valid for 1 hour)
        const presignedUrl = await env.DOWNLOADS.createPresignedUrl(fileName, {
          expiresIn: 3600, // 1 hour
          httpMethod: "GET",
        });

        return Response.json(
          {
            platform,
            fileName,
            downloadUrl: presignedUrl,
            fileSize: object.size,
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

    return new Response(null, { status: 404 });
  },
} satisfies ExportedHandler<Env>;
