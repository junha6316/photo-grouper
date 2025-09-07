import { Env, DOWNLOAD_FILES, getAvailableFile, corsHeaders } from "../_shared";

export const onRequest: PagesFunction<Env> = async (context) => {
  const { request, env } = context;
  const url = new URL(request.url);

  // Handle preflight requests
  if (request.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

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
      console.log(
        `File not found for platform ${platform}, looking for: ${baseName}`
      );
      return Response.json(
        { error: "Download file not found" },
        { status: 404, headers: corsHeaders }
      );
    }

    console.log(
      `Found file for ${platform}: ${fileInfo.fileName} (compressed: ${fileInfo.isCompressed})`
    );

    // Get file metadata
    const object = await env.DOWNLOADS.head(fileInfo.fileName);
    if (!object) {
      return Response.json(
        { error: "Download file not found" },
        { status: 404, headers: corsHeaders }
      );
    }

    // Generate download URL that points back to our API
    const downloadUrl = `${url.origin}/api/download-file?platform=${platform}`;

    const response = {
      platform,
      fileName: fileInfo.isCompressed ? baseName : fileInfo.fileName, // Return original name for user
      actualFileName: fileInfo.fileName, // Internal file name (may be compressed)
      downloadUrl,
      fileSize: object.size,
      isCompressed: fileInfo.isCompressed,
      compressionType: fileInfo.compressionType,
      expiresIn: 3600,
    };

    console.log(
      `Download info for ${platform}:`,
      JSON.stringify(response, null, 2)
    );

    return Response.json(response, { headers: corsHeaders });
  } catch (error) {
    console.error("Error generating download URL:", error);
    return Response.json(
      { error: "Failed to generate download URL" },
      { status: 500, headers: corsHeaders }
    );
  }
};
