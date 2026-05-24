import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type HubEntry = {
  user: string;
  snippet: string;
  score: number;
  action: string;
};

type AppContextValue = {
  threshold: number;
  setThreshold: (v: number) => void;
  hubHistory: HubEntry[];
  addHubEntry: (entry: HubEntry) => void;
};

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [threshold, setThreshold] = useState(0.5);
  const [hubHistory, setHubHistory] = useState<HubEntry[]>([]);

  const addHubEntry = useCallback((entry: HubEntry) => {
    setHubHistory((prev) => [entry, ...prev].slice(0, 100));
  }, []);

  const value = useMemo(
    () => ({
      threshold,
      setThreshold,
      hubHistory,
      addHubEntry,
    }),
    [threshold, hubHistory, addHubEntry]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
