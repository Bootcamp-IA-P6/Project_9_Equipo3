import { useEffect, useRef, useState } from "react";
import { predict } from "../api/client";
import type { PredictResponse } from "../types/api";

export function useDebouncedPredict(text: string, threshold: number, delayMs = 400) {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const trimmed = text.trim();
    if (!trimmed) {
      setResult(null);
      setError(null);
      setLoading(false);
      return;
    }

    const timer = setTimeout(() => {
      abortRef.current?.abort();
      setLoading(true);
      setError(null);
      predict(trimmed, threshold)
        .then(setResult)
        .catch((e: Error) => setError(e.message))
        .finally(() => setLoading(false));
    }, delayMs);

    return () => clearTimeout(timer);
  }, [text, threshold, delayMs]);

  return { result, loading, error };
}
