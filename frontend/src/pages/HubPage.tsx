import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useApp } from "../context/AppContext";
import { useTheme } from "../context/ThemeContext";
import { useI18n } from "../i18n/I18nContext";

function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  return (
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() ||
    fallback
  );
}

export function HubPage() {
  const { hubHistory, threshold } = useApp();
  const { theme } = useTheme();
  const { t } = useI18n();

  const colors = useMemo(
    () => ({
      safe: cssVar("--safe", "#2f8f3e"),
      toxic: cssVar("--toxic", "#c2331f"),
      accent: cssVar("--accent", "#d83d2c"),
      muted: cssVar("--muted", "#6b6256"),
    }),
    [theme]
  );

  const toxic = hubHistory.filter((h) => h.score >= threshold).length;
  const safe = hubHistory.length - toxic;
  const pieData = [
    { name: t.hub.safe, value: safe || 1 },
    { name: t.hub.toxic, value: toxic || 0 },
  ];
  const pieColors = [colors.safe, colors.toxic];

  const barData = hubHistory.slice(0, 12).map((h, i) => ({
    name: `#${i + 1}`,
    score: Math.round(h.score * 100),
  }));

  return (
    <div className="hub-page">
      <h1>{t.hub.title}</h1>
      <div className="hub-cards">
        <div className="hub-stat">
          <span className="stat-label">{t.hub.threshold}</span>
          <span className="stat-value">{threshold.toFixed(2)}</span>
        </div>
        <div className="hub-stat">
          <span className="stat-label">{t.hub.eventsLogged}</span>
          <span className="stat-value">{hubHistory.length}</span>
        </div>
        <div className="hub-stat">
          <span className="stat-label">{t.hub.toxicSession}</span>
          <span className="stat-value">{toxic}</span>
        </div>
      </div>

      <div className="hub-charts">
        <div className="chart-box">
          <h2>{t.hub.safeVsToxic}</h2>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={pieColors[i % pieColors.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-box">
          <h2>{t.hub.recentScores}</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData}>
              <XAxis dataKey="name" stroke={colors.muted} />
              <YAxis stroke={colors.muted} domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="score" fill={colors.accent} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="hub-table-wrap">
        <h2>{t.hub.recentActions}</h2>
        <table className="hub-table">
          <thead>
            <tr>
              <th>{t.hub.colUser}</th>
              <th>{t.hub.colComment}</th>
              <th>{t.hub.colScore}</th>
              <th>{t.hub.colAction}</th>
            </tr>
          </thead>
          <tbody>
            {hubHistory.length === 0 ? (
              <tr>
                <td colSpan={4}>{t.hub.emptyHistory}</td>
              </tr>
            ) : (
              hubHistory.map((h, i) => (
                <tr key={i}>
                  <td>{h.user}</td>
                  <td>{h.snippet}</td>
                  <td>{(h.score * 100).toFixed(0)}%</td>
                  <td>{h.action}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
