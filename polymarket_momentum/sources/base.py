from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import httpx
import pandas as pd


@dataclass
class Market:
    """Source-agnostic market metadata.

    - `id` / `slug`: venue-specific identifiers (Polymarket market id, Kalshi ticker).
    - `yes_id` / `no_id`: the identifiers you need to fetch orderbooks / prices
      for each side. For Polymarket these are CLOB token ids. For Kalshi these are
      the market ticker paired with a side label ("yes"/"no") — encoded as strings
      the source can parse back.
    """

    source: str
    id: str
    question: str
    slug: str
    yes_id: str
    no_id: str
    volume: float
    end_date: str | None
    closed: bool = False


@dataclass
class OrderBook:
    token_id: str
    # Each level: (price, size). Bids sorted high→low, asks sorted low→high.
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2


@dataclass
class Position:
    source: str
    asset: str
    market_id: str
    title: str
    slug: str
    outcome: str
    size: float
    avg_price: float
    current_price: float
    current_value: float
    initial_value: float
    cash_pnl: float
    percent_pnl: float
    end_date: str | None
    redeemable: bool


class FeeModel(Protocol):
    """Fee model: per-share fee for a fill at a given price on a given side."""

    name: str

    def per_share(self, *, side: str, price: float) -> float: ...


@dataclass
class FlatFee:
    """Flat fee as a fraction of notional (fee_rate * price per share)."""

    rate: float = 0.0
    name: str = "flat"

    def per_share(self, *, side: str, price: float) -> float:
        return self.rate * price


@dataclass
class KalshiFee:
    """Kalshi's round-up fee: rate * price * (1 - price) per contract.

    Actual Kalshi fees round up to the next cent per trade; this ignores rounding
    (fine for ranking / research, wrong at single-contract granularity).
    """

    rate: float = 0.07
    name: str = "kalshi"

    def per_share(self, *, side: str, price: float) -> float:
        return self.rate * price * (1.0 - price)


class MarketSource(Protocol):
    name: str
    fee_model: FeeModel

    def list_markets(
        self, *, min_volume: float = 0, limit: int = 500
    ) -> list[Market]: ...

    def get_price_history(
        self,
        market: Market,
        *,
        side: str = "yes",
        lookback_days: int | None = None,
        fidelity: int = 60,
        client: httpx.Client | None = None,
    ) -> pd.DataFrame: ...

    def get_order_book(
        self,
        market: Market,
        *,
        side: str = "yes",
        client: httpx.Client | None = None,
    ) -> OrderBook: ...

    def fetch_positions(self) -> list[Position]: ...

    def cache_prices(
        self,
        markets: Iterable[Market],
        out_dir: Path,
        *,
        side: str = "yes",
        lookback_days: int | None = None,
        fidelity: int = 60,
    ) -> list[str]: ...

    def write_market_metadata(self, markets: Iterable[Market], path: Path) -> None: ...
