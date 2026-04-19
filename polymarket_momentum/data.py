from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"


@dataclass
class Market:
    id: str
    question: str
    slug: str
    yes_token_id: str
    no_token_id: str
    volume: float
    end_date: str | None


def list_markets(
    *,
    closed: bool = False,
    min_volume: float = 10_000,
    limit: int = 500,
    include_closed: bool = False,
) -> list[Market]:
    out: list[Market] = _fetch_markets_page(
        closed=closed, min_volume=min_volume, limit=limit
    )
    if include_closed and not closed:
        more = _fetch_markets_page(
            closed=True, min_volume=min_volume, limit=limit
        )
        seen = {m.id for m in out}
        for m in more:
            if m.id not in seen:
                out.append(m)
                seen.add(m.id)
    return out[:limit]


def _fetch_markets_page(
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
    return Market(
        id=str(payload["id"]),
        question=payload.get("question", ""),
        slug=payload.get("slug", ""),
        yes_token_id=str(tokens[0]),
        no_token_id=str(tokens[1]),
        volume=volume,
        end_date=payload.get("endDate"),
    )


def get_price_history(
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


def get_price_history_chunked(
    token_id: str,
    *,
    lookback_days: int,
    fidelity: int = 60,
    chunk_days: int = 12,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    end_ts = int(time.time())
    start_ts = end_ts - lookback_days * 86_400
    frames: list[pd.DataFrame] = []
    cursor = start_ts

    owned = client is None
    c = client or httpx.Client(timeout=30)
    try:
        while cursor < end_ts:
            chunk_end = min(cursor + chunk_days * 86_400, end_ts)
            try:
                df = get_price_history(
                    token_id,
                    start_ts=cursor,
                    end_ts=chunk_end,
                    fidelity=fidelity,
                    client=c,
                )
            except httpx.HTTPError:
                df = pd.DataFrame(columns=["ts", "price"])
            if not df.empty:
                frames.append(df)
            cursor = chunk_end
    finally:
        if owned:
            c.close()

    if not frames:
        return pd.DataFrame(columns=["ts", "price"])
    return (
        pd.concat(frames)
        .drop_duplicates(subset=["ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )


def cache_prices(
    markets: Iterable[Market],
    out_dir: Path,
    *,
    fidelity: int = 60,
    interval: str = "1m",
    lookback_days: int | None = None,
    concurrency: int = 8,
    resolution_trim_hours: float = 24.0,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    markets = list(markets)
    if not markets:
        return []

    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    written: list[str] = []
    now_utc = pd.Timestamp.utcnow()

    with httpx.Client(timeout=30, limits=limits) as client:
        def fetch_one(market: Market) -> tuple[Market, pd.DataFrame | None]:
            try:
                if lookback_days is not None:
                    df = get_price_history_chunked(
                        market.yes_token_id,
                        lookback_days=lookback_days,
                        fidelity=fidelity,
                        client=client,
                    )
                else:
                    df = get_price_history(
                        market.yes_token_id,
                        interval=interval,
                        fidelity=fidelity,
                        client=client,
                    )
                return market, df
            except httpx.HTTPError as exc:
                print(f"skip {market.slug}: {exc}")
                return market, None

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(fetch_one, m) for m in markets]
            done = 0
            for fut in as_completed(futures):
                market, df = fut.result()
                done += 1
                if df is None or df.empty:
                    continue
                df = _trim_resolution(
                    df, market, now_utc=now_utc, trim_hours=resolution_trim_hours
                )
                if df.empty:
                    continue
                df.to_csv(out_dir / f"{market.id}.csv", index=False)
                written.append(market.id)
                if done % 25 == 0:
                    print(f"  [{done}/{len(markets)}] cached={len(written)}")

    return written


def _trim_resolution(
    df: pd.DataFrame,
    market: Market,
    *,
    now_utc: pd.Timestamp,
    trim_hours: float,
) -> pd.DataFrame:
    if trim_hours <= 0 or not market.end_date:
        return df
    try:
        end_ts = pd.Timestamp(market.end_date)
    except (ValueError, TypeError):
        return df
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    if end_ts >= now_utc:
        return df
    cutoff = end_ts - pd.Timedelta(hours=trim_hours)
    return df[df["ts"] < cutoff].reset_index(drop=True)


def write_market_metadata(markets: Iterable[Market], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(m) for m in markets]).to_csv(path, index=False)


@dataclass
class OrderBook:
    token_id: str
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


def get_order_book(token_id: str, *, client: httpx.Client | None = None) -> OrderBook:
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


def get_orderbook(token_id: str, *, client: httpx.Client | None = None) -> dict | None:
    """Return a compact top-of-book snapshot or None if either side is empty."""
    try:
        book = get_order_book(token_id, client=client)
    except httpx.HTTPError:
        return None
    bid = book.best_bid
    ask = book.best_ask
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    spread_bps = (ask - bid) / mid * 10_000.0
    return {
        "bid": float(bid),
        "ask": float(ask),
        "mid": float(mid),
        "spread_bps": float(spread_bps),
    }
