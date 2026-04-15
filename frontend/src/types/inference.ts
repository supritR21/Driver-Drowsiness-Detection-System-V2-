export type InferenceResponse = {
  status: string;
  session_id: string;
  sequence_length: number;
  score: number | null;
  level: "safe" | "soft" | "warning" | "danger" | null;
  prediction: "alert" | "drowsy" | "microsleep" | null;
  message: string | null;
  source: "model" | "heuristic" | null;
};