import { useI18n } from "../i18n/I18nContext";
import type { SuggestedVideo } from "../types/api";

type Props = {
  videos: SuggestedVideo[];
  activeId: string | null;
  loadingId: string | null;
  onSelect: (video: SuggestedVideo) => void;
};

export function SuggestedRail({ videos, activeId, loadingId, onSelect }: Props) {
  const { t } = useI18n();
  return (
    <aside className="suggested-rail">
      <p className="rail-title">{t.watch.upNext}</p>
      {videos.map((v) => (
        <button
          key={v.id}
          type="button"
          className={`suggested-card ${activeId === v.id ? "active" : ""}`}
          onClick={() => onSelect(v)}
          disabled={loadingId === v.id}
        >
          <img
            src={v.thumbnail_url}
            alt=""
            className="suggested-thumb"
            onError={(e) => {
              const img = e.currentTarget;
              if (!img.dataset.fallback) {
                img.dataset.fallback = "1";
                img.src = `https://i.ytimg.com/vi/${v.id}/hqdefault.jpg`;
              }
            }}
          />
          <div className="suggested-info">
            <p className="suggested-title">{v.title}</p>
            <p className="suggested-channel">{v.channel_title}</p>
            {!v.embeddable && <span className="embed-badge">{t.watch.externalOnly}</span>}
            {loadingId === v.id && <span className="loading-tag">{t.watch.loadingComments}</span>}
          </div>
        </button>
      ))}
    </aside>
  );
}
