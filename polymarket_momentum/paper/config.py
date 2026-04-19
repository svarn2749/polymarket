from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path


def _data_root() -> Path:
    # Railway-style volume mount takes priority, else fall back to repo-local data/.
    return Path("/data") if Path("/data").is_dir() else Path("data")


def _default_db_path() -> Path:
    return _data_root() / "paper.db"


def _default_markets_csv() -> Path:
    return _data_root() / "markets.csv"


def _default_spreads_csv() -> Path:
    return _data_root() / "spreads.csv"


@dataclass
class PaperConfig:
    source: str = "polymarket"

    # Universe selection
    markets_csv: Path = field(default_factory=_default_markets_csv)
    spreads_csv: Path = field(default_factory=_default_spreads_csv)
    universe_top_n: int = 30
    max_spread_bps: float = 400.0
    min_volume: float = 50_000.0  # for live fallback fetch on cold boot

    # Strategy (reversion is the validated winner)
    strategy: str = "reversion"
    lookback_hours: int = 24
    entry_threshold: float = 0.20

    # Execution
    fill_model: str = "realistic"  # "mid" | "realistic" | "half_spread"
    position_size_usd: float = 5.0  # per-market, fits $100 across ~20 markets
    min_trade_usd: float = 1.0      # skip tiny rebalances

    # Loop
    poll_interval_sec: int = 300     # 5 min
    startup_delay_sec: int = 5
    http_concurrency: int = 8
    refresh_interval_days: float = 7.0   # markets.csv + spreads.csv

    # Storage
    db_path: Path = field(default_factory=_default_db_path)

    # Misc
    enabled: bool = True  # set PAPER_ENABLED=0 to disable the poller

    @classmethod
    def from_env(cls) -> "PaperConfig":
        c = cls()
        for f in fields(cls):
            key = f"PAPER_{f.name.upper()}"
            raw = os.environ.get(key)
            if raw is None:
                continue
            current = getattr(c, f.name)
            if isinstance(current, Path):
                setattr(c, f.name, Path(raw))
            elif isinstance(current, bool):
                setattr(c, f.name, raw.lower() in {"1", "true", "yes", "on"})
            elif isinstance(current, int) and not isinstance(current, bool):
                setattr(c, f.name, int(raw))
            elif isinstance(current, float):
                setattr(c, f.name, float(raw))
            else:
                setattr(c, f.name, raw)
        return c
