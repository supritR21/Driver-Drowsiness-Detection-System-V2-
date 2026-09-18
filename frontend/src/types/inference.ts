export type InferenceLevel =
  | "safe"
  | "soft"
  | "warning"
  | "danger";

export type InferenceResponse = {
  status: string;

  session_id: string;

  sequence_length: number;

  score: number | null;

  level: InferenceLevel | null;

  prediction:
    | "alert"
    | "drowsy"
    | "microsleep"
    | null;

  message: string | null;

  source:
    | "model"
    | "heuristic"
    | "transformer"
    | null;

  raw_probability?: number | null;

  smoothed_probability?: number | null;

  decision_threshold?: number | null;

  vote_ratio?: number | null;

  sequence_ready?: boolean;

  source_samples?: number;

  coverage_seconds?: number;
};