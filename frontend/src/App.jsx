import { useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "";

export default function App() {
  const [theme, setTheme] = useState("dark");
  const [draft, setDraft] = useState("");
  const [comments, setComments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function submitComment() {
    const text = draft.trim();
    if (!text) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      setComments((prev) => [
        {
          id: crypto.randomUUID(),
          text,
          toxicScore: data.toxic_score,
          label: data.label,
          color: data.status_color,
          model: data.model_name,
        },
        ...prev,
      ]);
      setDraft("");
    } catch (e) {
      setError(e.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`app ${theme}`}>
      <button
        type="button"
        className="theme-toggle"
        onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      >
        {theme === "dark" ? "Light" : "Dark"} mode
      </button>

      <section className="video-section">
        <div className="video-placeholder">Sample watch page — video placeholder</div>
      </section>

      <section className="comments">
        <h2>Comments</h2>
        {error && <p className="error">{error}</p>}
        <div className="comment-form">
          <textarea
            placeholder="Add a comment..."
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            disabled={loading}
          />
          <div className="form-actions">
            <button type="button" onClick={submitComment} disabled={loading || !draft.trim()}>
              {loading ? "Analyzing…" : "Comment"}
            </button>
          </div>
        </div>

        <ul className="comment-list">
          {comments.map((c) => (
            <li key={c.id} className="comment-item">
              <div className="avatar" aria-hidden />
              <div className="comment-body">
                <p className="comment-text">{c.text}</p>
                <div className="comment-meta">
                  <span className={`badge ${c.color}`}>{c.label}</span>
                  <span className="score">Toxic: {c.toxic_score}%</span>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
