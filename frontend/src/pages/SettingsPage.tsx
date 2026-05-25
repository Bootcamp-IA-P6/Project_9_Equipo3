import { useEffect, useState } from "react";
import { getModelInfo, getModelsStatus, predict, setModel } from "../api/client";
import { useApp } from "../context/AppContext";
import type { ModelStatusEntry } from "../types/api";

export function SettingsPage() {
  const { threshold, setThreshold } = useApp();
  const [modelStatus, setModelStatus] = useState<ModelStatusEntry[]>([]);
  const [active, setActive] = useState("");
  const [testText, setTestText] = useState("You are an idiot");
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testError, setTestError] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [switching, setSwitching] = useState(false);

  const loadStatus = () => {
    getModelsStatus()
      .then((r) => {
        setModelStatus(r.models);
        setActive(r.active);
      })
      .catch(() => setMessage("Could not load model status"));
  };

  useEffect(() => {
    loadStatus();
  }, []);

  useEffect(() => {
    setTestResult(null);
    setTestError(null);
  }, [testText, threshold]);

  const switchModel = async (name: string) => {
    const entry = modelStatus.find((m) => m.name === name);
    if (entry && !entry.available) {
      setMessage(entry.reason ?? "Model unavailable");
      return;
    }
    setMessage(null);
    setSwitching(true);
    try {
      await setModel(name);
      setActive(name);
      setMessage(`Active model: ${name}`);
      const info = await getModelInfo();
      if (info.recommended_threshold != null) {
        setThreshold(info.recommended_threshold);
      }
      loadStatus();
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Failed to switch model");
      loadStatus();
    } finally {
      setSwitching(false);
    }
  };

  const runTest = async () => {
    setTesting(true);
    setTestResult(null);
    setTestError(null);
    try {
      const r = await predict(testText, threshold);
      setTestResult(`${r.status} — ${Math.round(r.probability * 100)}% toxic`);
    } catch (e) {
      setTestError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="settings-page">
      <h1>Settings</h1>
      <section className="settings-card">
        <h2>Active model</h2>
        <p className="production-model-note">
          Default: <strong>Meta-Feature Stacking (Production)</strong> (F1 0.805, gap 2.54%).
          Baselines: <strong>LR + TF-IDF</strong> (F1 0.758) and{" "}
          <strong>Frozen Toxic-BERT</strong> (F1 0.790, gap 0.16%).
        </p>
        <p className="hint">
          Production and frozen BERT need <code>uv sync --extra hf</code> (or Docker{" "}
          <code>INSTALL_HF=1</code>). LR baseline uses joblib only. First transformer load may
          download weights (~1 min).
        </p>
        {switching && (
          <p className="hint">Switching model… production may take up to a minute on first load.</p>
        )}
        <div className="model-list">
          {modelStatus.map((m) => (
            <label
              key={m.name}
              className={`model-option ${!m.available ? "model-unavailable" : ""}`}
            >
              <input
                type="radio"
                name="model"
                checked={active === m.name}
                disabled={!m.available || switching}
                onChange={() => void switchModel(m.name)}
              />
              <span>
                {m.name}
                {!m.available && m.reason && (
                  <span className="model-reason"> — {m.reason}</span>
                )}
              </span>
            </label>
          ))}
        </div>
        {message && (
          <p className={message.includes("Failed") || message.includes("Install") ? "error-text" : "settings-msg"}>
            {message}
          </p>
        )}
      </section>

      <section className="settings-card">
        <h2>Toxicity threshold</h2>
        <input
          type="range"
          min={0.1}
          max={0.9}
          step={0.05}
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
        />
        <p>{threshold.toFixed(2)} — comments at or above this probability are <strong>Toxic</strong>.</p>
      </section>

      <section className="settings-card">
        <h2>Quick test</h2>
        <textarea value={testText} onChange={(e) => setTestText(e.target.value)} rows={2} />
        <button
          type="button"
          className="btn-primary"
          disabled={testing || !testText.trim()}
          onClick={() => void runTest()}
        >
          {testing ? "Analyzing…" : "Analyze"}
        </button>
        {testResult && <p className="settings-msg">{testResult}</p>}
        {testError && <p className="error-text">{testError}</p>}
      </section>
    </div>
  );
}
