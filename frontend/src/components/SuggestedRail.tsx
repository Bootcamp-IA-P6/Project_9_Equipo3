import type { SuggestedVideo } from "../types/api";

type Props = {
  videos: SuggestedVideo[];
  activeId: string | null;
  loadingId: string | null;
  onSelect: (video: SuggestedVideo) => void;
};

export function SuggestedRail({ videos, activeId, loadingId, onSelect }: Props) {
  return (
    <aside className="suggested-rail">
      <p className="rail-title">Up next</p>
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
            {!v.embeddable && <span className="embed-badge">External only</span>}
            {loadingId === v.id && <span className="loading-tag">Loading comments…</span>}
          </div>
        </button>
      ))}
    </aside>
  );
}
