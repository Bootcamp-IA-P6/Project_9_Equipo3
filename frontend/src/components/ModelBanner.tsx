import { useEffect, useState } from "react";
import { getModelInfo } from "../api/client";
import { useI18n } from "../i18n/I18nContext";

export function ModelBanner() {
  const { t } = useI18n();
  const [banner, setBanner] = useState<string | null>(null);

  useEffect(() => {
    const fallback = t.modelBanner.current("Meta-Feature Stacking", "0.805", "2.54");
    getModelInfo()
      .then((info) => {
        const apiBanner = (info as { display_banner?: string }).display_banner;
        if (apiBanner) {
          setBanner(apiBanner);
          return;
        }
        if (info.name?.includes("Meta-Feature Stacking")) {
          setBanner(fallback);
          return;
        }
        setBanner(null);
      })
      .catch(() => {
        setBanner(fallback);
      });
  }, [t]);

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
