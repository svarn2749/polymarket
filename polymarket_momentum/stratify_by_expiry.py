"""Stratify backtest PnL by each bar's days-to-expiry.

For every market × bar in the backtest, assign a bucket based on how far
away resolution was at that timestamp, then aggregate PnL, turnover, and
hit rate by bucket. Answers: does the signal's edge concentrate in a
specific "days until resolution" regime?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest_market


BUCKETS = [
    ("<=1d", 0, 1),
    ("1-7d", 1, 7),
    ("7-30d", 7, 30),
    ("30-90d", 30, 90),
    ("90-365d", 90, 365),
    (">1y", 365, 10_000),
]


def _bucket_label(days: float) -> str:
    if np.isnan(days) or days < 0:
        return "expired"
    for label, lo, hi in BUCKETS:
        if lo <= days < hi:
            return label
    return "expired"


def stratify(
    data_dir: Path,
    markets_csv: Path,
    *,
    strategy: str,
    lookback_hours: int,
    entry_threshold: float,
    exit_threshold: float | None,
    fee_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    meta = pd.read_csv(markets_csv, dtype={"id": str})
    meta["end_date"] = pd.to_datetime(meta["end_date"], errors="coerce", utc=True)
    end_by_id = dict(zip(meta["id"].astype(str), meta["end_date"]))

    rows: list[dict] = []
    files = sorted(data_dir.glob("*.csv"))
    for i, path in enumerate(files, 1):
        mid = path.stem
        end_date = end_by_id.get(mid)
        if end_date is pd.NaT or end_date is None:
            continue
        df = pd.read_csv(path, parse_dates=["ts"])
        if len(df) < 96:
            continue
        prices = df.set_index("ts")["price"]
        result = backtest_market(
            prices,
            strategy=strategy,
            lookback_hours=lookback_hours,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        bars = result.bars
        if bars.empty:
            continue
        days_to_expiry = (end_date - bars.index).total_seconds() / 86_400.0
        bars = bars.assign(days_to_expiry=days_to_expiry, market_id=mid)
        bars["bucket"] = [_bucket_label(d) for d in bars["days_to_expiry"]]
        rows.append(bars)
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)} markets")

    if not rows:
        return pd.DataFrame()
    all_bars = pd.concat(rows, ignore_index=True)
    # Drop bars we can't score (e.g., empty bucket, or lookback burn-in zeros).
    all_bars = all_bars[all_bars["bucket"] != "expired"]

    grouped = all_bars.groupby("bucket", observed=True)
    turnover = all_bars.groupby("bucket", observed=True).apply(
        lambda g: float(g["position"].diff().abs().sum()), include_groups=False
    )
    summary = pd.DataFrame({
        "n_bars": grouped["pnl"].count(),
        "total_pnl": grouped["pnl"].sum(),
        "mean_pnl_per_bar": grouped["pnl"].mean(),
        "sharpe_annualized": grouped["pnl"].mean() / grouped["pnl"].std() * np.sqrt(8760),
        "hit_rate": grouped["pnl"].apply(lambda s: float((s > 0).mean())),
        "exposure": grouped["position"].apply(lambda s: float((s != 0).mean())),
        "turnover": turnover,
    })

    # Reorder by bucket semantic order, not alphabetic.
    order = [label for label, _, _ in BUCKETS if label in summary.index]
    summary = summary.reindex(order)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify backtest PnL by days-to-expiry bucket.")
    parser.add_argument("--data", type=Path, default=Path("data/prices"))
    parser.add_argument("--markets-csv", type=Path, default=Path("data/markets.csv"))
    parser.add_argument("--strategy", choices=["momentum", "reversion"], default="reversion")
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--entry", type=float, default=0.20)
    parser.add_argument("--exit", type=float, default=0.10, dest="exit_threshold",
                        help="exit threshold for hysteresis; 0 to disable")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=200.0)
    parser.add_argument("--out", type=Path, default=Path("data/stratify_by_expiry.csv"))
    args = parser.parse_args()

    exit_threshold = args.exit_threshold if args.exit_threshold > 0 else None

    summary = stratify(
        args.data, args.markets_csv,
        strategy=args.strategy,
        lookback_hours=args.lookback,
        entry_threshold=args.entry,
        exit_threshold=exit_threshold,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    if summary.empty:
        print("no data")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"\nstratification: strategy={args.strategy} lookback={args.lookback}h "
          f"entry={args.entry} exit={exit_threshold} slippage={args.slippage_bps}bps")
    print(summary.round(4))


if __name__ == "__main__":
    main()
