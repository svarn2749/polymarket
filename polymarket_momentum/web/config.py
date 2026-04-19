from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    source: str
    prices_dir: Path
    markets_csv: Path
    target_notional: float
    lookback_hours: int
    resample_rule: str
    min_abs_signal: float
    max_markets: int


def load_settings() -> Settings:
    return Settings(
        source=os.environ.get("POLYMARKET_SOURCE", "polymarket"),
        prices_dir=Path(os.environ.get("POLYMARKET_PRICES_DIR", "data/prices")),
        markets_csv=Path(os.environ.get("POLYMARKET_MARKETS_CSV", "data/markets.csv")),
        target_notional=float(os.environ.get("POLYMARKET_TARGET_NOTIONAL", "100")),
        lookback_hours=int(os.environ.get("POLYMARKET_LOOKBACK_HOURS", "24")),
        resample_rule=os.environ.get("POLYMARKET_RESAMPLE", "1h"),
        min_abs_signal=float(os.environ.get("POLYMARKET_MIN_SIGNAL", "0.02")),
        max_markets=int(os.environ.get("POLYMARKET_MAX_MARKETS", "50")),
    )
