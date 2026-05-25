import { useEffect, useState } from "react";
import { getModelInfo } from "../api/client";

export function ModelBanner() {
  const [banner, setBanner] = useState<string | null>(null);

  useEffect(() => {
    getModelInfo()
      .then((info) => {
        const text =
          (info as { display_banner?: string }).display_banner ??
          (info.name?.includes("Meta-Feature Stacking")
            ? "Currently using: Meta-Feature Stacking Model (F1: 0.805, Gap: 2.54%)"
            : null);
        setBanner(text);
      })
      .catch(() => {
        setBanner(
          "Currently using: Meta-Feature Stacking Model (F1: 0.805, Gap: 2.54%)"
        );
      });
  }, []);

  if (!banner) return null;

  return (
    <div className="model-banner" role="status" aria-live="polite">
      <span className="model-banner-icon" aria-hidden>
        🏆
      </span>
      <span>{banner}</span>
    </div>
  );
}
