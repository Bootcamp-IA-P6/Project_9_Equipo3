import { useEffect, useRef, useState } from "react";
import { predict } from "../api/client";
import type { PredictResponse } from "../types/api";

export function useDebouncedPredict(text: string, threshold: number, delayMs = 400) {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const seqRef = useRef(0);

  useEffect(() => {
    const trimmed = text.trim();
    if (!trimmed) {
      setResult(null);
      setError(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    const seq = ++seqRef.current;

    const timer = setTimeout(() => {
      setError(null);
      predict(trimmed, threshold)
        .then((res) => {
          if (seq === seqRef.current) setResult(res);
        })
        .catch((e: Error) => {
          if (seq === seqRef.current) setError(e.message);
        })
        .finally(() => {
          if (seq === seqRef.current) setLoading(false);
        });
    }, delayMs);

    return () => clearTimeout(timer);
  }, [text, threshold, delayMs]);

  return { result, loading, error };
}
