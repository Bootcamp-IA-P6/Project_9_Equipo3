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

export const TEAM_MEMBERS = [
  "Roberto Molero",
  "Mirae Kang",
  "Jonathan Brasales",
  "Andrés Torrez",
] as const;

export function randomTeamMember(): string {
  return TEAM_MEMBERS[Math.floor(Math.random() * TEAM_MEMBERS.length)];
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
  "electric", "atomic", "midnight", "golden", "silver", "crimson", "violet", "azure",
  "ancient", "feral", "tidal", "lunar", "solar", "ember", "frost", "iron",
  "velvet", "wicked", "groovy", "funky", "chill", "vibrant", "moody", "savage",
  "tropical", "alpine", "urban", "rogue", "stellar", "nomadic", "boreal", "obsidian",
];

const NOUNS = [
  "panda", "tiger", "falcon", "otter", "eagle", "wolf", "fox", "bear",
  "lynx", "shark", "raven", "viper", "phoenix", "dragon", "jaguar", "koala",
  "moose", "lion", "owl", "octopus", "rabbit", "hawk", "badger", "robin",
  "cosmonaut", "drifter", "ninja", "wizard", "rider", "pilot", "skater", "gamer",
  "barista", "hacker", "painter", "poet", "sailor", "ranger", "archer", "dancer",
  "chef", "cyclist", "writer", "drummer", "diver", "climber", "biker", "florist",
  "comet", "nebula", "nova", "pulsar", "quasar", "asteroid", "meteor", "orbiter",
  "knight", "monk", "scribe", "merchant", "wanderer", "voyager", "explorer", "pirate",
];

const SUFFIXES = [
  "", "", "", "", "",
  "_", "_x", "_yt", "_hd", "_real", "_official", "_tv", "_live", "_xd", "_v2",
  "99", "01", "07", "23", "42",
  "_irl", "_pro",
];

function hashString(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function mix(h: number, salt: number): number {
  let x = (h ^ Math.imul(salt + 1, 0x85ebca6b)) >>> 0;
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

/**
 * Generate a YouTube-style random username, deterministic per seed.
 * Three independent hash mixes pick adjective, noun, and suffix so that
 * seeds that share a prefix (e.g. `yt-VIDEOID-0`, `yt-VIDEOID-1`) still
 * produce visibly different usernames.
 */
export function randomUsername(seed: string | number | undefined | null): string {
  const str = seed == null ? "anon" : String(seed);
  const base = hashString(str);
  const adj = ADJECTIVES[mix(base, 1) % ADJECTIVES.length] ?? "anon";
  const noun = NOUNS[mix(base, 2) % NOUNS.length] ?? "user";
  const suffix = SUFFIXES[mix(base, 3) % SUFFIXES.length] ?? "";
  const needsNum = suffix === "" || suffix.endsWith("_");
  const num = needsNum ? String(mix(base, 4) % 1000) : "";
  return `${adj}_${noun}${suffix}${num}`;
}
