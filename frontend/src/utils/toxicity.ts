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
