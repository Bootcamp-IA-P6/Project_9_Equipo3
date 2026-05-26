import { useI18n } from "../i18n/I18nContext";
import { formatPct, toxicityColor } from "../utils/toxicity";
import type { CommentItem } from "../types/api";

type Props = { comment: CommentItem };

export function CommentRow({ comment }: Props) {
  const { t } = useI18n();
  const color = toxicityColor(comment.probability);
  return (
    <div className={`comment-row ${comment.is_toxic ? "toxic" : "safe"}`}>
      <div className={`avatar ${comment.is_toxic ? "toxic" : ""}`}>
        {comment.user.charAt(0).toUpperCase()}
      </div>
      <div className="comment-body">
        <div className="comment-meta">
          <span className="comment-user">@{comment.user}</span>
          <span className="comment-time">{comment.time}</span>
          <span className={`badge ${comment.is_toxic ? "badge-toxic" : "badge-safe"}`}>
            {comment.is_toxic ? t.badges.toxic : t.badges.safe}
          </span>
          <span className="comment-pct" style={{ color }}>
            {formatPct(comment.probability)}
          </span>
        </div>
        <p className="comment-text">{comment.text}</p>
        {comment.is_toxic && <p className="flagged">{t.watch.flagged}</p>}
        {comment.labels.length > 0 && (
          <p className="comment-labels">{comment.labels.join(" · ")}</p>
        )}
      </div>
    </div>
  );
}
