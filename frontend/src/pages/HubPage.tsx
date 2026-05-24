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

const COLORS = ["#2ba640", "#ff4e45"];

export function HubPage() {
  const { hubHistory, threshold } = useApp();

  const toxic = hubHistory.filter((h) => h.score >= threshold).length;
  const safe = hubHistory.length - toxic;
  const pieData = [
    { name: "Safe", value: safe || 1 },
    { name: "Toxic", value: toxic || 0 },
  ];

  const barData = hubHistory.slice(0, 12).map((h, i) => ({
    name: `#${i + 1}`,
    score: Math.round(h.score * 100),
  }));

  return (
    <div className="hub-page">
      <h1>Moderator Hub</h1>
      <div className="hub-cards">
        <div className="hub-stat">
          <span className="stat-label">Threshold</span>
          <span className="stat-value">{threshold.toFixed(2)}</span>
        </div>
        <div className="hub-stat">
          <span className="stat-label">Events logged</span>
          <span className="stat-value">{hubHistory.length}</span>
        </div>
        <div className="hub-stat">
          <span className="stat-label">Toxic (session)</span>
          <span className="stat-value">{toxic}</span>
        </div>
      </div>

      <div className="hub-charts">
        <div className="chart-box">
          <h2>Safe vs Toxic</h2>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-box">
          <h2>Recent scores (%)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData}>
              <XAxis dataKey="name" stroke="#aaa" />
              <YAxis stroke="#aaa" domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="score" fill="#3ea6ff" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="hub-table-wrap">
        <h2>Recent actions</h2>
        <table className="hub-table">
          <thead>
            <tr>
              <th>User</th>
              <th>Comment</th>
              <th>Score</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {hubHistory.length === 0 ? (
              <tr>
                <td colSpan={4}>Post comments on the Watch page to populate history.</td>
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
