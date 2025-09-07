import { Env, DOWNLOAD_FILES, getAvailableFile } from "../_shared";

export const onRequest: PagesFunction<Env> = async (context) => {
  const { request, env } = context;
  const url = new URL(request.url);

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
    if (fileInfo.isCompressed && fileInfo.compressionType === "gzip") {
      // For gzip files, let browser automatically decompress
      headers.set("Content-Type", "application/octet-stream");
      headers.set("Content-Encoding", "gzip");
      headers.set(
        "Content-Disposition",
        `attachment; filename="${baseName}"` // Use original filename
      );
    } else if (fileInfo.isCompressed) {
      // For other compression types (like xz), download as-is
      headers.set("Content-Type", "application/octet-stream");
      headers.set(
        "Content-Disposition",
        `attachment; filename="${fileInfo.fileName}"` // Use compressed filename
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
};
