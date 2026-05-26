import { useEffect, useState } from "react";
import { getModelInfo, getModelsStatus, predict, setModel } from "../api/client";
import { useApp } from "../context/AppContext";
import { useI18n } from "../i18n/I18nContext";
import type { ModelStatusEntry } from "../types/api";

export function SettingsPage() {
  const { threshold, setThreshold } = useApp();
  const { t } = useI18n();
  const [modelStatus, setModelStatus] = useState<ModelStatusEntry[]>([]);
  const [active, setActive] = useState("");
  const [testText, setTestText] = useState<string>(() => t.settings.defaultTestText);
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testError, setTestError] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageIsError, setMessageIsError] = useState(false);
  const [switching, setSwitching] = useState(false);

  const loadStatus = () => {
    getModelsStatus()
      .then((r) => {
        setModelStatus(r.models);
        setActive(r.active);
      })
      .catch(() => {
        setMessage(t.settings.couldNotLoadStatus);
        setMessageIsError(true);
      });
  };

  useEffect(() => {
    loadStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setTestResult(null);
    setTestError(null);
  }, [testText, threshold]);

  const switchModel = async (name: string) => {
    const entry = modelStatus.find((m) => m.name === name);
    if (entry && !entry.available) {
      setMessage(entry.reason ?? "");
      setMessageIsError(true);
      return;
    }
    setMessage(null);
    setMessageIsError(false);
    setSwitching(true);
    try {
      await setModel(name);
      setActive(name);
      setMessage(t.settings.activeModelMsg(name));
      setMessageIsError(false);
      const info = await getModelInfo();
      if (info.recommended_threshold != null) {
        setThreshold(info.recommended_threshold);
      }
      loadStatus();
    } catch (e) {
      setMessage(e instanceof Error ? e.message : t.settings.failedSwitch);
      setMessageIsError(true);
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
      setTestResult(
        t.settings.testResult(
          r.is_toxic ? t.badges.toxic : t.badges.safe,
          String(Math.round(r.probability * 100))
        )
      );
    } catch (e) {
      setTestError(e instanceof Error ? e.message : t.settings.analysisFailed);
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="settings-page">
      <h1>{t.settings.title}</h1>
      <section className="settings-card">
        <h2>{t.settings.activeModel}</h2>
        <p className="production-model-note">{t.settings.productionNote("0.805", "2.54")}</p>
        <p className="production-model-note">{t.settings.baselinesNote("0.758", "0.790", "0.16")}</p>
        <p className="hint">{t.settings.installHint}</p>
        {switching && <p className="hint">{t.settings.switching}</p>}
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
          <p className={messageIsError ? "error-text" : "settings-msg"}>
            {message}
          </p>
        )}
      </section>

      <section className="settings-card">
        <h2>{t.settings.thresholdTitle}</h2>
        <input
          type="range"
          min={0.1}
          max={0.9}
          step={0.05}
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
        />
        <p>
          {threshold.toFixed(2)} — {t.settings.thresholdNote}
        </p>
      </section>

      <section className="settings-card">
        <h2>{t.settings.quickTest}</h2>
        <textarea value={testText} onChange={(e) => setTestText(e.target.value)} rows={2} />
        <button
          type="button"
          className="btn-primary"
          disabled={testing || !testText.trim()}
          onClick={() => void runTest()}
        >
          {testing ? t.settings.analyzing : t.settings.analyze}
        </button>
        {testResult && <p className="settings-msg">{testResult}</p>}
        {testError && <p className="error-text">{testError}</p>}
      </section>
    </div>
  );
}
