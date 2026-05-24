import type {
  ModelStatusEntry,
  PredictResponse,
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

export function predict(text: string, threshold: number) {
  return request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({ text, threshold }),
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

export function getModelsStatus() {
  return request<{ models: ModelStatusEntry[]; active: string }>("/models/status");
}

export function setModel(name: string) {
  return request<{ message: string; model: string }>(`/model/${encodeURIComponent(name)}`, {
    method: "PUT",
  });
}

export function getSuggestedVideos() {
  return request<{ videos: SuggestedVideo[]; max_comments: number }>("/videos/suggested");
}

export function getModelInfo() {
  return request<{
    name: string;
    description: string;
    predictions_served: number;
  }>("/model-info");
}
