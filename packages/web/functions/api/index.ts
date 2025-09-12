import { corsHeaders } from "../_shared";

export const onRequest: PagesFunction = async (context) => {
  return new Response(
    JSON.stringify({
      name: "Photo Grouper Download API",
      version: "1.0.0",
    }),
    { headers: corsHeaders, status: 200 }
  );
};
