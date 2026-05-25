export function toxicityColor(probability: number): string {
  const pct = probability * 100;
  if (pct >= 70) return "#ff4e45";
  if (pct >= 40) return "#f5a623";
  return "#2ba640";
}

export function formatPct(probability: number): string {
  return `${Math.round(probability * 100)}%`;
}

export function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

const RTF = new Intl.RelativeTimeFormat("es", { numeric: "auto" });

const TIME_UNITS: Array<[Intl.RelativeTimeFormatUnit, number]> = [
  ["year", 60 * 60 * 24 * 365],
  ["month", 60 * 60 * 24 * 30],
  ["day", 60 * 60 * 24],
  ["hour", 60 * 60],
  ["minute", 60],
  ["second", 1],
];

export function relativeTime(input: string | number | Date): string {
  const date = input instanceof Date ? input : new Date(input);
  const seconds = (date.getTime() - Date.now()) / 1000;
  if (!Number.isFinite(seconds)) return "";
  for (const [unit, secondsInUnit] of TIME_UNITS) {
    if (Math.abs(seconds) >= secondsInUnit || unit === "second") {
      return RTF.format(Math.round(seconds / secondsInUnit), unit);
    }
  }
  return "";
}

export function truncate(text: string, max = 140): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1).trimEnd()}…`;
}

const ADJECTIVES = [
  "happy", "lucky", "wild", "calm", "bright", "swift", "quiet", "lazy",
  "brave", "fuzzy", "shiny", "sleepy", "quick", "noble", "loud", "humble",
  "stormy", "sunny", "rainy", "frosty", "spicy", "salty", "sweet", "crazy",
  "mighty", "silent", "epic", "cosmic", "rusty", "neon", "wandering", "lone",
];

const NOUNS = [
  "panda", "tiger", "falcon", "otter", "eagle", "wolf", "fox", "bear",
  "lynx", "shark", "raven", "viper", "phoenix", "dragon", "jaguar", "koala",
  "moose", "lion", "owl", "octopus", "rabbit", "hawk", "badger", "robin",
  "cosmonaut", "drifter", "ninja", "wizard", "rider", "pilot", "skater", "gamer",
];

const SUFFIXES = ["", "_", "_", "_yt", "_hd", "_real", "99", "01", "_xd", "_v2", "_official", ""];

/**
 * Generate a YouTube-style random username, deterministic per seed.
 */
export function randomUsername(seed: string | number | undefined | null): string {
  const str = seed == null ? "anon" : String(seed);
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) >>> 0;
  }
  const adj = ADJECTIVES[hash % ADJECTIVES.length] ?? "anon";
  const noun = NOUNS[Math.floor(hash / 32) % NOUNS.length] ?? "user";
  const suffix = SUFFIXES[Math.floor(hash / 1024) % SUFFIXES.length] ?? "";
  const needsNum = suffix === "" || suffix.endsWith("_");
  const num = needsNum ? String(hash % 1000) : "";
  return `${adj}_${noun}${suffix}${num}`;
}
