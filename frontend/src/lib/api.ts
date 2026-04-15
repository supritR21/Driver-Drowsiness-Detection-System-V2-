import { InferenceResponse } from "@/types/inference";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

export async function sendFrame(
  sessionId: string,
  frameBase64: string
): Promise<InferenceResponse> {
  const res = await fetch(`${API_BASE}/inference/frame`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      frame_base64: frameBase64,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Failed to send frame");
  }

  return res.json();
}