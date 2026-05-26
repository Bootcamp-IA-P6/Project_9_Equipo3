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
      .catch(() => setFetchError("Could not load suggested videos"));
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
        user: "you",
        text,
        time: "just now",
        is_toxic: analysis.is_toxic,
        probability: analysis.probability,
        labels: analysis.labels,
        source: "manual",
      };
      setSessionComments((prev) => [...prev, item]);
      addHubEntry({
        user: "@you",
        snippet: text.slice(0, 45),
        score: analysis.probability,
        action: analysis.is_toxic ? "Posted (toxic)" : "Approved",
      });
      setDraft("");
      void refreshRecent(activeVideo?.id);
    } finally {
      setPosting(false);
    }
  }, [draft, posting, result, threshold, addHubEntry, refreshRecent, activeVideo?.id]);

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
          time: "from YouTube",
          is_toxic: r.is_toxic,
          probability: r.probability,
          labels: r.labels,
          source: "youtube" as const,
        }))
      );
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : "Failed to load comments");
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
                <span className="player-fallback-cta">Watch on YouTube (embedding blocked)</span>
              </a>
            ) : (
              <iframe
                className="player-iframe"
                src={`https://www.youtube.com/embed/${
                  activeVideo?.id ?? DEFAULT_EMBED_VIDEO_ID
                }?rel=0${activeVideo ? "&autoplay=1" : ""}`}
                title={activeVideo?.title ?? "YouTube video player"}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                referrerPolicy="strict-origin-when-cross-origin"
                allowFullScreen
                loading="lazy"
              />
            )}
          </div>

          <h1 className="video-title">
            {activeVideo?.title ?? "Watch and moderate comments"}
          </h1>
          <p className="video-meta">
            {activeVideo
              ? activeVideo.channel_title
              : "Choose a video from Up next to load and score its comments"}
          </p>

          {activeVideo && isPlaceholderTitle(activeVideo.title, activeVideo.id) && (
            <p className="info-banner">
              Demo metadata — add <code>YOUTUBE_API_KEY</code> to <code>.env</code> for real titles.
            </p>
          )}

          <div className="channel-row">
            <div className="channel-avatar">{channelInitial}</div>
            <div>
              <p className="channel-name">{activeVideo?.channel_title ?? "YouTube"}</p>
              {activeVideo && <p className="video-meta">Suggested video</p>}
            </div>
          </div>

          <div className="comments-header">
            <span>
              {totalComments} comments
              {toxicManual + toxicYt > 0 && (
                <span className="toxic-count"> · {toxicManual + toxicYt} toxic detected</span>
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
                  if (!loading) void handlePost();
                }
              }}
              placeholder="Add a comment…"
              rows={3}
              aria-label="Write a comment"
            />
            {draft.trim() && (
              <div className="live-analysis">
                <span>{loading ? "Analyzing…" : "Live score"}</span>
                {result && (
                  <>
                    <span className={`badge ${result.is_toxic ? "badge-toxic" : "badge-safe"}`}>
                      {result.status}
                    </span>
                    <span style={{ color: toxicityColor(result.probability) }}>
                      Toxicity: {formatPct(result.probability)}
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
                Cancel
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={() => void handlePost()}
                disabled={posting || !draft.trim() || loading}
              >
                {posting ? "Posting…" : loading ? "Analyzing…" : "Comment"}
              </button>
            </div>
          </div>

          {fetchError && <p className="error-banner">{fetchError}</p>}

          {demoBanner && !dismissDemoBanner && (
            <div className="info-banner dismissible">
              <span>
                Using demo comments — add <code>YOUTUBE_API_KEY</code> to <code>.env</code> for real
                YouTube threads.
              </span>
              <button
                type="button"
                className="btn-dismiss"
                onClick={() => setDismissDemoBanner(true)}
                aria-label="Dismiss"
              >
                ×
              </button>
            </div>
          )}

          {loadingVideoId && youtubeComments.length === 0 && (
            <p className="loading-comments">Loading comments…</p>
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
            <p className="loading-comments">Loading recent comments…</p>
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
