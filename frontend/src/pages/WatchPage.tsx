import { useCallback, useEffect, useState } from "react";
import {
  getSuggestedVideos,
  listPredictions,
  predict,
  predictVideo,
} from "../api/client";
import { CommentRow } from "../components/CommentRow";
import { SuggestedRail } from "../components/SuggestedRail";
import { useApp } from "../context/AppContext";
import { useDebouncedPredict } from "../hooks/useDebouncedPredict";
import { useI18n } from "../i18n/I18nContext";
import type {
  CommentItem,
  PredictionRecord,
  SuggestedVideo,
} from "../types/api";
import {
  formatPct,
  newId,
  randomUsername,
  relativeTime,
  toxicityColor,
  truncate,
} from "../utils/toxicity";

const DEFAULT_EMBED_VIDEO_ID = "A1uxPRUgimk";

function isPlaceholderTitle(title: string, id: string): boolean {
  return title === `Video ${id}`;
}

export function WatchPage() {
  const { t } = useI18n();
  const { threshold, addHubEntry } = useApp();
  const [draft, setDraft] = useState("");
  const [sessionComments, setSessionComments] = useState<CommentItem[]>([]);
  const [suggested, setSuggested] = useState<SuggestedVideo[]>([]);
  const [maxComments, setMaxComments] = useState(15);
  const [activeVideo, setActiveVideo] = useState<SuggestedVideo | null>(null);
  const [youtubeComments, setYoutubeComments] = useState<CommentItem[]>([]);
  const [loadingVideoId, setLoadingVideoId] = useState<string | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [demoBanner, setDemoBanner] = useState(false);
  const [dismissDemoBanner, setDismissDemoBanner] = useState(false);
  const [posting, setPosting] = useState(false);
  const [recentActivity, setRecentActivity] = useState<PredictionRecord[]>([]);
  const [recentLoading, setRecentLoading] = useState(false);

  const { result, loading, error } = useDebouncedPredict(draft, threshold);

  const refreshRecent = useCallback(async (videoId?: string) => {
    setRecentLoading(true);
    try {
      const res = await listPredictions(videoId, 20);
      setRecentActivity(Array.isArray(res?.predictions) ? res.predictions : []);
    } catch {
      // Degrade gracefully if endpoint is missing or DB not configured
      setRecentActivity([]);
    } finally {
      setRecentLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshRecent(activeVideo?.id);
  }, [activeVideo?.id, refreshRecent]);

  useEffect(() => {
    getSuggestedVideos()
      .then((r) => {
        setSuggested(r.videos);
        setMaxComments(r.max_comments);
        // Auto-load first video so the user sees comments on initial render.
        if (r.videos.length > 0) {
          void loadVideo(r.videos[0]);
        }
      })
      .catch(() => setFetchError(t.watch.couldNotLoadVideos));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handlePost = useCallback(async () => {
    const text = draft.trim();
    if (!text || posting) return;
    setPosting(true);
    try {
      const analysis = result ?? (await predict(text, threshold));
      const item: CommentItem = {
        id: newId(),
        user: t.watch.you,
        text,
        time: t.watch.justNow,
        is_toxic: analysis.is_toxic,
        probability: analysis.probability,
        labels: analysis.labels,
        source: "manual",
      };
      setSessionComments((prev) => [...prev, item]);
      addHubEntry({
        user: `@${t.watch.you}`,
        snippet: text.slice(0, 45),
        score: analysis.probability,
        action: analysis.is_toxic
          ? `${t.watch.posted} (${t.badges.toxic.toLowerCase()})`
          : t.badges.safe,
      });
      setDraft("");
      void refreshRecent(activeVideo?.id);
    } finally {
      setPosting(false);
    }
  }, [draft, posting, result, threshold, addHubEntry, refreshRecent, activeVideo?.id, t]);

  const loadVideo = async (video: SuggestedVideo) => {
    setActiveVideo(video);
    setYoutubeComments([]);
    setSessionComments([]);
    setFetchError(null);
    setDismissDemoBanner(false);
    setLoadingVideoId(video.id);
    try {
      const res = await predictVideo(video.watch_url, maxComments, threshold);
      setDemoBanner(res.source === "demo");
      setYoutubeComments(
        res.results.map((r, i) => ({
          id: `yt-${video.id}-${i}`,
          user: randomUsername(`yt-${video.id}-${i}`),
          text: r.text,
          time: t.watch.fromYoutube,
          is_toxic: r.is_toxic,
          probability: r.probability,
          labels: r.labels,
          source: "youtube" as const,
        }))
      );
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : t.watch.failedToLoadComments);
      setYoutubeComments([]);
      setDemoBanner(false);
    } finally {
      setLoadingVideoId(null);
    }
  };

  const toxicManual = sessionComments.filter((c) => c.is_toxic).length;
  const toxicYt = youtubeComments.filter((c) => c.is_toxic).length;
  const totalComments = sessionComments.length + youtubeComments.length;

  const channelInitial = activeVideo?.channel_title?.charAt(0).toUpperCase() ?? "Y";

  return (
    <div className="watch-page">
      <div className="watch-grid">
        <section className="primary-column">
          <div className="staged-player">
            {activeVideo && !activeVideo.embeddable ? (
              <a
                className="player-fallback"
                href={activeVideo.watch_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                <img
                  src={activeVideo.thumbnail_url}
                  alt=""
                  className="player-fallback-thumb"
                />
                <span className="player-fallback-cta">{t.watch.watchOnYoutube}</span>
              </a>
            ) : (
              <iframe
                className="player-iframe"
                src={`https://www.youtube.com/embed/${
                  activeVideo?.id ?? DEFAULT_EMBED_VIDEO_ID
                }?rel=0`}
                title={activeVideo?.title ?? "YouTube video player"}
                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                referrerPolicy="strict-origin-when-cross-origin"
                allowFullScreen
                loading="lazy"
              />
            )}
          </div>

          <h1 className="video-title">
            {activeVideo?.title ?? t.watch.defaultTitle}
          </h1>
          <p className="video-meta">
            {activeVideo
              ? activeVideo.channel_title
              : t.watch.defaultMeta}
          </p>

          {activeVideo && isPlaceholderTitle(activeVideo.title, activeVideo.id) && (
            <p className="info-banner">{t.watch.placeholderTitleBanner}</p>
          )}

          <div className="channel-row">
            <div className="channel-avatar">{channelInitial}</div>
            <div>
              <p className="channel-name">{activeVideo?.channel_title ?? t.watch.channelFallback}</p>
              {activeVideo && <p className="video-meta">{t.watch.suggestedVideo}</p>}
            </div>
          </div>

          <div className="comments-header">
            <span>
              {t.watch.commentsCount(totalComments)}
              {toxicManual + toxicYt > 0 && (
                <span className="toxic-count">{t.watch.toxicDetected(toxicManual + toxicYt)}</span>
              )}
            </span>
          </div>

          <div className="comment-compose">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handlePost();
                }
              }}
              placeholder={t.watch.composePlaceholder}
              rows={3}
              aria-label={t.watch.composeAriaLabel}
            />
            {draft.trim() && (
              <div className="live-analysis">
                <span>{loading ? t.watch.analyzing : t.watch.liveScore}</span>
                {result && (
                  <>
                    <span className={`badge ${result.is_toxic ? "badge-toxic" : "badge-safe"}`}>
                      {result.is_toxic ? t.badges.toxic : t.badges.safe}
                    </span>
                    <span style={{ color: toxicityColor(result.probability) }}>
                      {`${t.watch.toxicity}: ${formatPct(result.probability)}`}
                    </span>
                  </>
                )}
                {error && <span className="error-text">{error}</span>}
              </div>
            )}
            <div className="compose-actions">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setDraft("")}
                disabled={posting}
              >
                {t.watch.cancel}
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={() => void handlePost()}
                disabled={posting || !draft.trim()}
              >
                {posting ? t.watch.analyzing : t.watch.comment}
              </button>
            </div>
          </div>

          {fetchError && <p className="error-banner">{fetchError}</p>}

          {demoBanner && !dismissDemoBanner && (
            <div className="info-banner dismissible">
              <span>{t.watch.demoBanner}</span>
              <button
                type="button"
                className="btn-dismiss"
                onClick={() => setDismissDemoBanner(true)}
                aria-label={t.watch.dismiss}
              >
                ×
              </button>
            </div>
          )}

          {loadingVideoId && youtubeComments.length === 0 && (
            <p className="loading-comments">{t.watch.loadingComments}</p>
          )}

          <div className="comment-list">
            {/* Local Supabase comments — always at the top (newest first) */}
            {recentActivity.map((rec, idx) => (
              <CommentRow
                key={`recent-${rec.id ?? idx}`}
                comment={{
                  id: `recent-${rec.id ?? idx}`,
                  user: randomUsername(`supa-${rec.id ?? idx}`),
                  text: truncate(rec.text, 140),
                  time: relativeTime(rec.created_at),
                  is_toxic: rec.is_toxic,
                  probability: rec.probability,
                  labels: rec.labels ?? [],
                  source: "recent",
                }}
              />
            ))}
            {/* Current session (just posted) */}
            {[...sessionComments].reverse().map((c) => (
              <CommentRow key={c.id} comment={c} />
            ))}
            {/* YouTube fetched comments — below */}
            {youtubeComments.map((c) => (
              <CommentRow key={c.id} comment={c} />
            ))}
          </div>

          {recentLoading && recentActivity.length === 0 && youtubeComments.length === 0 && (
            <p className="loading-comments">{t.watch.loadingRecent}</p>
          )}
        </section>

        <SuggestedRail
          videos={suggested}
          activeId={activeVideo?.id ?? null}
          loadingId={loadingVideoId}
          onSelect={(v) => void loadVideo(v)}
        />
      </div>
    </div>
  );
}
