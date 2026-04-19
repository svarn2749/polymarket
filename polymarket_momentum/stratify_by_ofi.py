"""Backtest: does order-flow imbalance (OFI) predict future price moves?

Pulls historical trades from Polymarket's data-api for each market, aggregates
to hourly OFI = (buy_vol - sell_vol) / total_vol, then joins against cached
hourly price history. For each hourly bar, compute the price change over the
next N hours. Bucket by OFI quantile and report mean forward-return per bucket.

If the top OFI quintile's forward return is meaningfully positive and the
bottom quintile's is meaningfully negative, OFI has predictive power and is
worth wiring into the live decision logic.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


TRADES_URL = "https://data-api.polymarket.com/trades"
MARKET_URL = "https://gamma-api.polymarket.com/markets"


def fetch_condition_id(market_id: str, *, client: httpx.Client) -> str | None:
    try:
        r = client.get(f"{MARKET_URL}/{market_id}")
        if r.status_code != 200:
            return None
        return r.json().get("conditionId")
    except httpx.HTTPError:
        return None


def fetch_trades(
    condition_id: str,
    yes_id: str,
    *,
    client: httpx.Client,
    page_size: int = 1000,
    max_pages: int = 5,
) -> pd.DataFrame:
    rows: list[dict] = []
    for page in range(max_pages):
        resp = client.get(
            TRADES_URL,
            params={"market": condition_id, "limit": page_size, "offset": page * page_size},
        )
        if resp.status_code != 200:
            break
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
    if not rows:
        return pd.DataFrame(columns=["ts", "side", "size", "price"])

    # The /trades response for a conditionId contains trades on both YES and
    # NO assets. Keep only YES-side so OFI is interpretable in YES space.
    df = pd.DataFrame(rows)
    df = df[df["asset"].astype(str) == str(yes_id)]
    if df.empty:
        return pd.DataFrame(columns=["ts", "side", "size", "price"])
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["size"] = df["size"].astype(float)
    df["price"] = df["price"].astype(float)
    return df[["ts", "side", "size", "price"]].sort_values("ts").reset_index(drop=True)


def hourly_ofi(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["ofi", "volume"])
    signed = np.where(trades["side"] == "BUY", trades["size"], -trades["size"])
    df = trades.assign(signed=signed, total=trades["size"]).set_index("ts")
    agg = df[["signed", "total"]].resample("1h").sum()
    agg["ofi"] = np.where(agg["total"] > 0, agg["signed"] / agg["total"], np.nan)
    agg["volume"] = agg["total"]
    return agg[["ofi", "volume"]]


def per_market_panel(
    market_id: str,
    yes_id: str,
    prices_csv: Path,
    *,
    client: httpx.Client,
    horizons_hours: list[int],
) -> pd.DataFrame:
    price_df = pd.read_csv(prices_csv, parse_dates=["ts"])
    if price_df.empty or len(price_df) < 48:
        return pd.DataFrame()
    prices = price_df.set_index("ts")["price"].resample("1h").last().ffill()

    condition_id = fetch_condition_id(market_id, client=client)
    if not condition_id:
        return pd.DataFrame()
    trades = fetch_trades(condition_id, yes_id, client=client)
    ofi = hourly_ofi(trades)
    if ofi.empty:
        return pd.DataFrame()

    # Align — use intersection of price hours and OFI hours
    joined = pd.concat([prices.rename("price"), ofi], axis=1, join="inner").dropna(subset=["price"])
    if joined.empty:
        return pd.DataFrame()

    for h in horizons_hours:
        joined[f"fwd_return_{h}h"] = joined["price"].shift(-h) - joined["price"]
    joined["market_id"] = market_id
    return joined.reset_index()


def load_yes_ids(markets_csv: Path) -> dict[str, str]:
    df = pd.read_csv(markets_csv, dtype=str)
    return dict(zip(df["id"].astype(str), df["yes_id"].astype(str)))


def run(
    prices_dir: Path,
    markets_csv: Path,
    *,
    max_markets: int,
    concurrency: int,
    horizons_hours: list[int],
) -> pd.DataFrame:
    yes_ids = load_yes_ids(markets_csv)
    price_files = sorted(prices_dir.glob("*.csv"))[:max_markets]
    print(f"fetching trades for {len(price_files)} markets (concurrency={concurrency})")

    frames: list[pd.DataFrame] = []
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    with httpx.Client(timeout=30, limits=limits) as client:
        def task(path: Path):
            mid = path.stem
            yid = yes_ids.get(mid)
            if not yid:
                return mid, None
            try:
                return mid, per_market_panel(mid, yid, path, client=client, horizons_hours=horizons_hours)
            except Exception as exc:
                print(f"  {mid} error: {exc}")
                return mid, None

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(task, p) for p in price_files]
            done = 0
            for fut in as_completed(futs):
                done += 1
                mid, panel = fut.result()
                if panel is None or panel.empty:
                    continue
                frames.append(panel)
                if done % 25 == 0:
                    print(f"  [{done}/{len(price_files)}] panels={len(frames)}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


OFI_BUCKETS = [
    ("heavy-sell",  -1.01, -0.30),
    ("sell",        -0.30, -0.10),
    ("neutral",     -0.10,  0.10),
    ("buy",          0.10,  0.30),
    ("heavy-buy",    0.30,  1.01),
]


def summarize(
    all_bars: pd.DataFrame,
    horizons_hours: list[int],
    *,
    min_volume: float = 20.0,
) -> pd.DataFrame:
    """Bucket by OFI level → report forward return per bucket.

    Filters out hours with tiny volume since those produce degenerate
    OFI values (±1 from a single trade).
    """
    if all_bars.empty:
        return pd.DataFrame()
    rows = []
    for h in horizons_hours:
        col = f"fwd_return_{h}h"
        df = all_bars[["ofi", "volume", col]].dropna()
        df = df[df["volume"] >= min_volume]
        if len(df) < 100:
            continue
        for label, lo, hi in OFI_BUCKETS:
            g = df[(df["ofi"] >= lo) & (df["ofi"] < hi)]
            if len(g) < 50:
                continue
            rows.append({
                "horizon_h": h,
                "ofi_bucket": label,
                "n": len(g),
                "mean_fwd_return": float(g[col].mean()),
                "median_fwd_return": float(g[col].median()),
                "mean_ofi": float(g["ofi"].mean()),
                "mean_volume": float(g["volume"].mean()),
            })
    return pd.DataFrame(rows)


def simulate_reversion_with_filter(
    panel: pd.DataFrame,
    *,
    lookback_hours: int,
    entry_threshold: float,
    exit_threshold: float,
    ofi_threshold: float | None,
    slippage_bps: float,
    position_size_usd: float,
) -> dict:
    """Backtest the reversion strategy on a price+OFI panel.

    Applies the hysteresis position logic per market. When `ofi_threshold`
    is set, also gates new entries: skip if the live-imputed imbalance (we
    use `ofi` from the panel as the proxy) opposes the desired direction by
    more than `ofi_threshold`.
    """
    from polymarket_momentum.strategy import (
        momentum_signal,
        sized_position_hysteresis,
    )

    total_pnl = 0.0
    total_fee = 0.0
    bar_pnls: list[float] = []
    n_entries = 0
    n_entries_blocked = 0
    n_markets = 0

    for mid, g in panel.groupby("market_id"):
        g = g.sort_values("ts").copy()
        prices = g["price"].reset_index(drop=True)
        ofi = g["ofi"].reset_index(drop=True)
        if len(prices) < lookback_hours + 2:
            continue
        n_markets += 1

        sig = momentum_signal(prices, lookback_hours)
        pos_raw = sized_position_hysteresis(
            sig, entry_threshold=entry_threshold, exit_threshold=exit_threshold,
        )
        # reversion: desired direction is the opposite sign of the signal
        pos_rev = -pos_raw

        # Apply OFI filter to NEW entries only
        if ofi_threshold is not None:
            pos_rev = pos_rev.copy()
            prev = pos_rev.shift(1).fillna(0.0)
            is_new_entry = (prev == 0) & (pos_rev != 0)
            # Block entry when OFI opposes desired direction by more than threshold
            # (long desired but OFI < -t) or (short desired but OFI > +t)
            adverse = (
                ((pos_rev > 0) & (ofi < -ofi_threshold)) |
                ((pos_rev < 0) & (ofi > ofi_threshold))
            )
            blocked_mask = is_new_entry & adverse.fillna(False)
            n_entries_blocked += int(blocked_mask.sum())
            pos_rev = pos_rev.where(~blocked_mask, 0.0)
            # Forward-fill the blocked-zero so we don't keep hunting for entry
            # across every adverse bar (matches the stateless "no entry this bar" model)

        # Position size in shares (directional): direction * $notional / price
        shares = pos_rev * position_size_usd / prices.replace(0, np.nan)
        price_change = prices.diff().fillna(0.0)
        gross_pnl = shares.shift(1).fillna(0.0) * price_change
        turnover = shares.diff().abs().fillna(0.0)
        fee = turnover * prices * (slippage_bps / 10_000.0)
        net = gross_pnl - fee

        total_pnl += float(net.sum())
        total_fee += float(fee.sum())
        bar_pnls.extend(net.tolist())
        entries = (pos_rev.diff().fillna(pos_rev).abs() > 0) & (pos_rev != 0)
        n_entries += int(entries.sum())

    if not bar_pnls:
        return {}
    import numpy as _np
    arr = _np.array(bar_pnls)
    sharpe = (arr.mean() / arr.std() * _np.sqrt(8760)) if arr.std() > 0 else 0.0
    return {
        "n_markets": n_markets,
        "n_entries": n_entries,
        "n_entries_blocked": n_entries_blocked,
        "total_pnl": float(arr.sum()),
        "total_fee": total_fee,
        "sharpe": float(sharpe),
        "mean_pnl_per_market": float(arr.sum() / n_markets) if n_markets else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest OFI predictive power.")
    p.add_argument("--prices-dir", type=Path, default=Path("data/prices"))
    p.add_argument("--markets-csv", type=Path, default=Path("data/markets.csv"))
    p.add_argument("--max-markets", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    p.add_argument("--out", type=Path, default=Path("data/ofi_backtest.csv"))
    p.add_argument("--sim-filter", action="store_true",
                   help="Also simulate reversion-with-OFI-filter vs baseline")
    p.add_argument("--lookback-hours", type=int, default=24)
    p.add_argument("--entry-threshold", type=float, default=0.20)
    p.add_argument("--exit-threshold", type=float, default=0.10)
    p.add_argument("--slippage-bps", type=float, default=200.0)
    p.add_argument("--position-size-usd", type=float, default=5.0)
    p.add_argument("--ofi-filter-thresholds", type=float, nargs="+",
                   default=[0.2, 0.3, 0.5], help="try multiple OFI thresholds")
    args = p.parse_args()

    all_bars = run(
        args.prices_dir, args.markets_csv,
        max_markets=args.max_markets,
        concurrency=args.concurrency,
        horizons_hours=args.horizons,
    )
    if all_bars.empty:
        print("no data")
        return

    print(f"\ntotal (market, hour) rows: {len(all_bars):,}")
    print(f"OFI range: [{all_bars['ofi'].min():+.3f}, {all_bars['ofi'].max():+.3f}]  mean={all_bars['ofi'].mean():+.3f}  std={all_bars['ofi'].std():.3f}")

    summary = summarize(all_bars, args.horizons)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n=== forward return by OFI bucket ===")
    print(summary.round(5).to_string(index=False))

    if args.sim_filter:
        print("\n=== reversion w/ OFI entry filter ===")
        kwargs = dict(
            lookback_hours=args.lookback_hours,
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
            slippage_bps=args.slippage_bps,
            position_size_usd=args.position_size_usd,
        )
        baseline = simulate_reversion_with_filter(all_bars, ofi_threshold=None, **kwargs)
        print("baseline (no OFI filter):", baseline)
        for t in args.ofi_filter_thresholds:
            res = simulate_reversion_with_filter(all_bars, ofi_threshold=t, **kwargs)
            print(f"OFI filter t={t}:", res)


if __name__ == "__main__":
    main()
