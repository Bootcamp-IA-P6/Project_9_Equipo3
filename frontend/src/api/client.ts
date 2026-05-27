import type {
  ModelStatusEntry,
  PredictResponse,
  PredictionsListResponse,
  SuggestedVideo,
  VideoResponse,
} from "../types/api";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

function parseApiError(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object") return fallback;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg);
        }
        return String(item);
      })
      .join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(parseApiError(err, res.statusText));
  }
  return res.json() as Promise<T>;
}

type PredictOptions = {
  videoId?: string;
  author?: string;
  persist?: boolean;
};

export function predict(text: string, threshold: number, options: PredictOptions = {}) {
  return request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({
      text,
      threshold,
      video_id: options.videoId,
      author: options.author,
      persist: options.persist ?? true,
    }),
  });
}

export function predictVideo(url: string, maxComments: number, threshold: number) {
  return request<VideoResponse>("/predict-video", {
    method: "POST",
    body: JSON.stringify({ url, max_comments: maxComments, threshold }),
  });
}

export function getModels() {
  return request<{ available: string[]; active: string }>("/models");
}

export async function getModelsStatus() {
  try {
    return await request<{ models: ModelStatusEntry[]; active: string }>("/models/status");
  } catch (e) {
    if (e instanceof Error && e.message.toLowerCase().includes("not found")) {
      const legacy = await getModels();
      return {
        active: legacy.active,
        models: legacy.available.map((name) => ({
          name,
          available: true,
          reason: null,
          type: "unknown",
        })),
      };
    }
    throw e;
  }
}

export function setModel(name: string) {
  return request<{ message: string; model: string }>("/models/select", {
    method: "POST",
    body: JSON.stringify({ model_name: name }),
  });
}

export function getSuggestedVideos() {
  return request<{ videos: SuggestedVideo[]; max_comments: number }>("/videos/suggested");
}

export function listPredictions(videoId?: string, limit = 200, source?: string) {
  const params = new URLSearchParams();
  if (videoId) params.set("video_id", videoId);
  if (source) params.set("source", source);
  params.set("limit", String(limit));
  return request<PredictionsListResponse>(`/predictions?${params.toString()}`);
}

export function getModelInfo() {
  return request<{
    name: string;
    description: string;
    predictions_served: number;
    display_banner?: string | null;
    train_test_gap_pp?: number | null;
    recommended_threshold?: number | null;
    accuracy?: string;
  }>("/model-info");
}
