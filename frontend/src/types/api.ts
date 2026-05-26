export type PredictResponse = {
  text: string;
  is_toxic: boolean;
  probability: number;
  status: "Safe" | "Toxic";
  mode: "binary";
  labels: string[];
  model_used: string;
  latency_ms: number;
};

export type SuggestedVideo = {
  id: string;
  title: string;
  channel_title: string;
  thumbnail_url: string;
  watch_url: string;
  embeddable: boolean;
};

export type VideoResponse = {
  video_url: string;
  total_fetched: number;
  toxic_count: number;
  toxic_rate: number;
  results: PredictResponse[];
  source: "youtube" | "demo" | "unavailable";
  reason?: string;
};

export type ModelStatusEntry = {
  name: string;
  available: boolean;
  reason: string | null;
  type: string;
};

export type CommentItem = {
  id: string;
  user: string;
  text: string;
  time: string;
  is_toxic: boolean;
  probability: number;
  labels: string[];
  source: "manual" | "youtube" | "recent";
};

export type PredictionRecord = {
  id: string | number;
  text: string;
  is_toxic: boolean;
  probability: number;
  video_id?: string | null;
  created_at: string;
  labels?: string[];
};

export type PredictionsListResponse = {
  predictions: PredictionRecord[];
  total?: number;
};
