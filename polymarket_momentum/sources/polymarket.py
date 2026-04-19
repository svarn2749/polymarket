from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd

from .base import FlatFee, Market, OrderBook, Position

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
DATA_API_POSITIONS = "https://data-api.polymarket.com/positions"

SOURCE_NAME = "polymarket"


class PolymarketSource:
    name = SOURCE_NAME

    def __init__(self, *, fee_rate: float = 0.0) -> None:
        self.fee_model = FlatFee(rate=fee_rate)

    # ---- markets ----

    def list_markets(
        self,
        *,
        min_volume: float = 10_000,
        limit: int = 500,
        include_closed: bool = False,
    ) -> list[Market]:
        out = _list_markets_page(closed=False, min_volume=min_volume, limit=limit)
        if include_closed:
            more = _list_markets_page(closed=True, min_volume=min_volume, limit=limit)
            seen = {m.id for m in out}
            for m in more:
                if m.id not in seen:
                    out.append(m)
                    seen.add(m.id)
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
        interval: str = "1m",
        client: httpx.Client | None = None,
    ) -> pd.DataFrame:
        token_id = market.yes_id if side == "yes" else market.no_id
        if lookback_days is not None:
            return _get_price_history_chunked(
                token_id, lookback_days=lookback_days, fidelity=fidelity, client=client
            )
        return _get_price_history(
            token_id, interval=interval, fidelity=fidelity, client=client
        )

    def cache_prices(
        self,
        markets: Iterable[Market],
        out_dir: Path,
        *,
        side: str = "yes",
        lookback_days: int | None = None,
        fidelity: int = 60,
        interval: str = "1m",
        concurrency: int = 8,
        resolution_trim_hours: float = 0.0,
    ) -> list[str]:
        """Cache price history to CSV.

        `resolution_trim_hours`: for closed markets, drop the trailing N hours
        to avoid the resolution jump (price pinning at 0 or 1) polluting the
        backtest signal.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        markets = list(markets)
        if not markets:
            return []
        written: list[str] = []
        now_utc = pd.Timestamp.utcnow()
        limits = httpx.Limits(
            max_keepalive_connections=concurrency,
            max_connections=concurrency * 2,
        )

        with httpx.Client(timeout=30, limits=limits) as client:
            def _fetch_and_trim(market: Market) -> pd.DataFrame | None:
                try:
                    df = self.get_price_history(
                        market,
                        side=side,
                        lookback_days=lookback_days,
                        fidelity=fidelity,
                        interval=interval,
                        client=client,
                    )
                except httpx.HTTPError as exc:
                    print(f"skip {market.slug}: {exc}")
                    return None
                if df.empty:
                    return df
                if resolution_trim_hours > 0 and _end_in_past(market, now_utc):
                    cutoff = df["ts"].max() - pd.Timedelta(hours=resolution_trim_hours)
                    df = df[df["ts"] <= cutoff]
                return df

            def _task(market: Market) -> tuple[Market, pd.DataFrame | None]:
                return market, _fetch_and_trim(market)

            if concurrency <= 1:
                for m in markets:
                    _, df = _task(m)
                    if df is not None and not df.empty:
                        df.to_csv(out_dir / f"{m.id}.csv", index=False)
                        written.append(m.id)
                return written

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_task, m) for m in markets]
                done = 0
                for fut in as_completed(futures):
                    market, df = fut.result()
                    done += 1
                    if df is None or df.empty:
                        continue
                    df.to_csv(out_dir / f"{market.id}.csv", index=False)
                    written.append(market.id)
                    if done % 50 == 0:
                        print(f"  [{done}/{len(markets)}] cached={len(written)}")
        return written

    # ---- orderbook ----

    def get_order_book(
        self, market: Market, *, side: str = "yes", client: httpx.Client | None = None
    ) -> OrderBook:
        token_id = market.yes_id if side == "yes" else market.no_id
        return get_order_book_by_token(token_id, client=client)

    # ---- positions ----

    def fetch_positions(
        self, *, wallet: str | None = None, size_threshold: float = 1.0
    ) -> list[Position]:
        addr = wallet or os.environ.get("POLYMARKET_WALLET")
        if not addr:
            raise RuntimeError("POLYMARKET_WALLET env var not set")
        with httpx.Client(timeout=20) as client:
            resp = client.get(
                DATA_API_POSITIONS,
                params={"user": addr, "sizeThreshold": size_threshold},
            )
            resp.raise_for_status()
            payload = resp.json()
        return [_parse_position(row) for row in payload or []]


# ---- module-level helpers (shared by source + legacy callers in tests) ----


def _list_markets_page(
    *, closed: bool, min_volume: float, limit: int
) -> list[Market]:
    out: list[Market] = []
    offset = 0
    page_size = 100
    with httpx.Client(timeout=30) as client:
        while len(out) < limit:
            params: dict[str, str | int] = {
                "closed": str(closed).lower(),
                "archived": "false",
                "limit": page_size,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }
            if not closed:
                params["active"] = "true"
            resp = client.get(GAMMA_URL, params=params)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for m in batch:
                market = _parse_market(m, min_volume=min_volume)
                if market is not None:
                    out.append(market)
            offset += page_size
            if len(batch) < page_size:
                break
    return out[:limit]


def _parse_market(payload: dict, *, min_volume: float) -> Market | None:
    tok_raw = payload.get("clobTokenIds")
    if not tok_raw:
        return None
    try:
        tokens = json.loads(tok_raw) if isinstance(tok_raw, str) else tok_raw
    except json.JSONDecodeError:
        return None
    if len(tokens) != 2:
        return None
    volume = float(payload.get("volume") or 0)
    if volume < min_volume:
        return None
    events = payload.get("events") or []
    event_ticker = str(events[0].get("ticker", "")) if events else ""
    return Market(
        source=SOURCE_NAME,
        id=str(payload["id"]),
        question=payload.get("question", ""),
        slug=payload.get("slug", ""),
        yes_id=str(tokens[0]),
        no_id=str(tokens[1]),
        volume=volume,
        end_date=payload.get("endDate"),
        closed=bool(payload.get("closed")),
        event_ticker=event_ticker,
        fee_type=str(payload.get("feeType") or ""),
    )


def _get_price_history(
    token_id: str,
    *,
    interval: str | None = "1m",
    start_ts: int | None = None,
    end_ts: int | None = None,
    fidelity: int = 60,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    params: dict[str, str | int] = {"market": token_id, "fidelity": fidelity}
    if start_ts is not None and end_ts is not None:
        params["startTs"] = start_ts
        params["endTs"] = end_ts
    else:
        params["interval"] = interval or "max"
    owned = client is None
    c = client or httpx.Client(timeout=30)
    try:
        resp = c.get(CLOB_PRICES_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if owned:
            c.close()
    history = payload.get("history", [])
    if not history:
        return pd.DataFrame(columns=["ts", "price"])
    df = pd.DataFrame(history).rename(columns={"t": "ts", "p": "price"})
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    return df.sort_values("ts").reset_index(drop=True)


def _get_price_history_chunked(
    token_id: str,
    *,
    lookback_days: int,
    fidelity: int = 60,
    chunk_days: int = 12,
    sleep_s: float = 0.05,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    end_ts = int(time.time())
    start_ts = end_ts - lookback_days * 86_400
    frames: list[pd.DataFrame] = []
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + chunk_days * 86_400, end_ts)
        try:
            df = _get_price_history(
                token_id,
                start_ts=cursor,
                end_ts=chunk_end,
                fidelity=fidelity,
                client=client,
            )
        except httpx.HTTPError:
            df = pd.DataFrame(columns=["ts", "price"])
        if not df.empty:
            frames.append(df)
        cursor = chunk_end
        if sleep_s > 0:
            time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame(columns=["ts", "price"])
    return (
        pd.concat(frames)
        .drop_duplicates(subset=["ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )


def _end_in_past(market: Market, now_utc: pd.Timestamp) -> bool:
    if market.closed:
        return True
    if not market.end_date:
        return False
    try:
        end_ts = pd.Timestamp(market.end_date)
    except (ValueError, TypeError):
        return False
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    return end_ts < now_utc


def get_order_book_by_token(
    token_id: str, *, client: httpx.Client | None = None
) -> OrderBook:
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=15)
    try:
        resp = client.get(CLOB_BOOK_URL, params={"token_id": token_id})
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if owns_client:
            client.close()

    def _levels(raw: list[dict]) -> list[tuple[float, float]]:
        out = [(float(lv["price"]), float(lv["size"])) for lv in raw or []]
        return [(p, s) for p, s in out if s > 0]

    bids = sorted(_levels(payload.get("bids", [])), key=lambda x: -x[0])
    asks = sorted(_levels(payload.get("asks", [])), key=lambda x: x[0])
    return OrderBook(token_id=token_id, bids=bids, asks=asks)


def _parse_position(row: dict) -> Position:
    return Position(
        source=SOURCE_NAME,
        asset=str(row.get("asset", "")),
        market_id=str(row.get("conditionId", "")),
        title=row.get("title", ""),
        slug=row.get("slug", ""),
        outcome=row.get("outcome", ""),
        size=float(row.get("size") or 0),
        avg_price=float(row.get("avgPrice") or 0),
        current_price=float(row.get("curPrice") or 0),
        current_value=float(row.get("currentValue") or 0),
        initial_value=float(row.get("initialValue") or 0),
        cash_pnl=float(row.get("cashPnl") or 0),
        percent_pnl=float(row.get("percentPnl") or 0),
        end_date=row.get("endDate"),
        redeemable=bool(row.get("redeemable") or False),
    )
