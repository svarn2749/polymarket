from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd

from .base import KalshiFee, Market, OrderBook, Position

API_ROOT = "https://api.elections.kalshi.com/trade-api/v2"
MARKETS_URL = f"{API_ROOT}/markets"

SOURCE_NAME = "kalshi"


class KalshiSource:
    name = SOURCE_NAME

    def __init__(self, *, fee_rate: float = 0.07) -> None:
        self.fee_model = KalshiFee(rate=fee_rate)

    # ---- markets ----

    def list_markets(
        self,
        *,
        min_volume: float = 1_000,
        limit: int = 500,
        status: str = "open",
    ) -> list[Market]:
        out: list[Market] = []
        cursor: str | None = None
        with httpx.Client(timeout=30) as client:
            while len(out) < limit:
                params: dict[str, str | int] = {
                    "status": status,
                    "limit": min(200, limit - len(out)),
                }
                if cursor:
                    params["cursor"] = cursor
                resp = client.get(MARKETS_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()
                for row in payload.get("markets", []):
                    market = _parse_market(row, min_volume=min_volume)
                    if market is not None:
                        out.append(market)
                cursor = payload.get("cursor") or None
                if not cursor:
                    break
        return out[:limit]

    def write_market_metadata(self, markets: Iterable[Market], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(m) for m in markets]).to_csv(path, index=False)

    # ---- prices ----

    def get_price_history(
        self,
        market: Market,
        *,
        side: str = "yes",
        lookback_days: int | None = None,
        fidelity: int = 60,
    ) -> pd.DataFrame:
        ticker = market.id
        end_ts = int(time.time())
        start_ts = end_ts - (lookback_days or 30) * 86_400
        url = f"{API_ROOT}/series/{market.slug}/markets/{ticker}/candlesticks"
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": fidelity,
        }
        with httpx.Client(timeout=30) as client:
            try:
                resp = client.get(url, params=params)
                resp.raise_for_status()
            except httpx.HTTPError:
                return pd.DataFrame(columns=["ts", "price"])
            payload = resp.json()
        candles = payload.get("candlesticks", [])
        if not candles:
            return pd.DataFrame(columns=["ts", "price"])
        rows = []
        for c in candles:
            price_cents = (c.get("yes_bid", {}).get("close") or 0) / 2 + (
                c.get("yes_ask", {}).get("close") or 0
            ) / 2
            if price_cents <= 0:
                continue
            price = price_cents / 100.0
            if side == "no":
                price = 1.0 - price
            rows.append({"ts": c["end_period_ts"], "price": price})
        if not rows:
            return pd.DataFrame(columns=["ts", "price"])
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df.sort_values("ts").reset_index(drop=True)

    def cache_prices(
        self,
        markets: Iterable[Market],
        out_dir: Path,
        *,
        side: str = "yes",
        lookback_days: int | None = 30,
        fidelity: int = 60,
        concurrency: int = 8,
    ) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out_dir.mkdir(parents=True, exist_ok=True)
        markets = list(markets)
        written: list[str] = []

        def _task(market: Market) -> tuple[Market, pd.DataFrame]:
            df = self.get_price_history(
                market, side=side, lookback_days=lookback_days, fidelity=fidelity
            )
            return market, df

        if concurrency <= 1:
            for m in markets:
                _, df = _task(m)
                if not df.empty:
                    df.to_csv(out_dir / f"{m.id}.csv", index=False)
                    written.append(m.id)
            return written

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_task, m) for m in markets]
            for fut in as_completed(futures):
                m, df = fut.result()
                if df.empty:
                    continue
                df.to_csv(out_dir / f"{m.id}.csv", index=False)
                written.append(m.id)
        return written

    # ---- orderbook ----

    def get_order_book(
        self, market: Market, *, side: str = "yes", client: httpx.Client | None = None
    ) -> OrderBook:
        ticker = market.id
        url = f"{API_ROOT}/markets/{ticker}/orderbook"
        owns_client = client is None
        if owns_client:
            client = httpx.Client(timeout=15)
        try:
            resp = client.get(url)
            resp.raise_for_status()
            body = resp.json()
        finally:
            if owns_client:
                client.close()

        # Kalshi response wrapper migrated: `orderbook_fp` with dollar-denominated
        # string decimals in `yes_dollars` / `no_dollars`. Fall back to the older
        # `orderbook` / `yes` / `no` shape for resilience.
        payload = body.get("orderbook_fp") or body.get("orderbook") or {}
        yes_raw = payload.get("yes_dollars") or payload.get("yes") or []
        no_raw = payload.get("no_dollars") or payload.get("no") or []

        # `yes` bids: buyers willing to pay P for Yes. "yes" list is bids on yes.
        # "no" bids at P are equivalent to yes asks at (1 - P).
        yes_levels = [(_as_float(p), _as_float(s)) for p, s in yes_raw]
        no_levels = [(_as_float(p), _as_float(s)) for p, s in no_raw]
        yes_levels = [(p, s) for p, s in yes_levels if p > 0 and s > 0]
        no_levels = [(p, s) for p, s in no_levels if p > 0 and s > 0]

        if side == "yes":
            bids = sorted(yes_levels, key=lambda x: -x[0])
            asks = sorted(
                [(1.0 - p, s) for p, s in no_levels], key=lambda x: x[0]
            )
        else:
            bids = sorted(no_levels, key=lambda x: -x[0])
            asks = sorted(
                [(1.0 - p, s) for p, s in yes_levels], key=lambda x: x[0]
            )

        return OrderBook(token_id=f"{ticker}:{side}", bids=bids, asks=asks)

    # ---- positions ----

    def fetch_positions(self) -> list[Position]:
        raise NotImplementedError(
            "Kalshi positions require authenticated API access "
            "(KALSHI_API_KEY_ID + KALSHI_API_PRIVATE_KEY). Not yet wired."
        )


def _parse_market(payload: dict, *, min_volume: float) -> Market | None:
    ticker = payload.get("ticker")
    event_ticker = payload.get("event_ticker")
    if not ticker or not event_ticker:
        return None
    # Kalshi moved volume onto `volume_fp` (string decimal). Older `volume` is gone.
    volume = _as_float(payload.get("volume_fp") or payload.get("volume"))
    if volume < min_volume:
        return None
    return Market(
        source=SOURCE_NAME,
        id=str(ticker),
        question=payload.get("title") or payload.get("subtitle") or "",
        # Kalshi has no "slug"; event_ticker is the series-level identifier
        # (also used by the candlesticks endpoint).
        slug=str(event_ticker),
        yes_id=f"{ticker}:yes",
        no_id=f"{ticker}:no",
        volume=volume,
        end_date=payload.get("close_time"),
    )


def _as_float(x) -> float:
    if x is None or x == "":
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0
