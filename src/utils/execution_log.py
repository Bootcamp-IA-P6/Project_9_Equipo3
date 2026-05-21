"""Append structured lines to logs/execution.log (AGENTS.md quality standards)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path("logs/execution.log")


def log_event(
    message: str,
    *,
    phase: str | None = None,
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one timestamped line to the execution log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prefix = f"[{ts}]"
    if phase:
        prefix += f" [phase={phase}]"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{prefix} {message}\n")
