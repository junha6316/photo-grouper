// Note: You would need to compile your TS into JS and output it as a `_worker.js` file. We do not read `_worker.ts`
import {
  corsHeaders,
  DOWNLOAD_FILES,
  type Env,
  getAvailableFile,
} from "./_worker-utils";

export default {
  async fetch(request: Request, env: Env) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }
    if (!url.pathname.startsWith("/api/download")) {
      return new Response(null, {
        headers: corsHeaders,
        status: 200,
      });
    }

    try {
      // Handle CORS preflight requests
      if (request.method === "OPTIONS") {
        return new Response(null, {
          headers: corsHeaders,
          status: 200,
        });
      }

      const url = new URL(request.url);
      const platform = url.searchParams.get("platform");

      // Validate platform parameter
      if (
        !platform ||
        !Object.keys(DOWNLOAD_FILES).includes(platform.toLowerCase())
      ) {
        return new Response(
          JSON.stringify({
            error:
              "Invalid platform. Supported platforms: macos, windows, linux",
          }),
          {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
            status: 400,
          }
        );
      }

      const getAvailableFileResult = await getAvailableFile(
        env.DOWNLOADS,
        DOWNLOAD_FILES[platform.toLowerCase() as keyof typeof DOWNLOAD_FILES]
      );

      if (!getAvailableFileResult) {
        return new Response(JSON.stringify({ error: "File not found" }), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          status: 404,
        });
      }

      return new Response(
        JSON.stringify({
          platform: platform.toLowerCase(),
          fileName: getAvailableFileResult.fileName,
          downloadUrl: getAvailableFileResult.downloadUrl,
          fileSize: getAvailableFileResult.fileSize,
          expiresIn: getAvailableFileResult.expiresIn,
        }),
        {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          status: 200,
        }
      );
    } catch (error) {
      console.error("Download API error:", error);
      return new Response(JSON.stringify({ error: "Internal server error" }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
        status: 500,
      });
    }
  },
};
