import { corsHeaders } from "../_shared";

export const onRequest: PagesFunction = async () => {
  return Response.json(
    {
      name: "Photo Grouper Download API",
      version: "1.0.0",
    },
    { headers: corsHeaders }
  );
};
