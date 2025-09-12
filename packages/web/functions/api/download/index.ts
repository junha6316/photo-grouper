import {
  corsHeaders,
  DOWNLOAD_FILES,
  type Env,
  getAvailableFile,
} from "./_shared";
// Types for Pages Functions
type PagesFunction<Env = unknown> = (context: {
  request: Request;
  env: Env;
  params: Record<string, string>;
  waitUntil: (promise: Promise<any>) => void;
  next: (input?: Request | string, init?: RequestInit) => Promise<Response>;
}) => Response | Promise<Response>;

export const onRequest: PagesFunction = async (
  context: EventContext<Env, any, Record<string, unknown>>
) => {
  try {
    // Handle CORS preflight requests
    if (context.request.method === "OPTIONS") {
      return new Response(null, {
        headers: corsHeaders,
        status: 200,
      });
    }

    const url = new URL(context.request.url);
    const platform = url.searchParams.get("platform");

    // Validate platform parameter
    if (
      !platform ||
      !Object.keys(DOWNLOAD_FILES).includes(platform.toLowerCase())
    ) {
      return new Response(
        JSON.stringify({
          error: "Invalid platform. Supported platforms: macos, windows, linux",
        }),
        { 
          headers: { ...corsHeaders, "Content-Type": "application/json" }, 
          status: 400 
        }
      );
    }

    const getAvailableFileResult = await getAvailableFile(
      context.env.DOWNLOADS,
      DOWNLOAD_FILES[platform.toLowerCase() as keyof typeof DOWNLOAD_FILES]
    );

    if (!getAvailableFileResult) {
      return new Response(
        JSON.stringify({ error: "File not found" }), 
        {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          status: 404,
        }
      );
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
        status: 200 
      }
    );
  } catch (error) {
    console.error("Download API error:", error);
    return new Response(
      JSON.stringify({ error: "Internal server error" }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
        status: 500,
      }
    );
  }
};
