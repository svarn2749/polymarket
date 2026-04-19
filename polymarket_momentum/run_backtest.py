from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import backtest_directory


def load_spreads(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, dtype={"market_id": str})
    return dict(zip(df["market_id"].astype(str), df["spread_bps"].astype(float)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run momentum backtest on cached prices.")
    parser.add_argument("--data", type=Path, default=Path("data/prices"))
    parser.add_argument("--lookback", type=int, default=24, help="lookback in hourly bars")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Round-trip slippage assumption in bps of price, additive to --fee-bps",
    )
    parser.add_argument(
        "--spreads-csv",
        type=Path,
        default=None,
        help="CSV with market_id,spread_bps columns; overrides --slippage-bps per market",
    )
    parser.add_argument(
        "--strategy",
        choices=["momentum", "reversion"],
        default="momentum",
    )
    parser.add_argument("--out", type=Path, default=Path("data/backtest_summary.csv"))
    args = parser.parse_args()

    spreads = load_spreads(args.spreads_csv) if args.spreads_csv else None

    summary = backtest_directory(
        args.data,
        lookback_hours=args.lookback,
        entry_threshold=args.threshold,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        strategy=args.strategy,
        spreads=spreads,
    )
    if summary.empty:
        print(f"no usable data in {args.data}")
        return

    summary = summary.sort_values("total_pnl", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(f"backtested {len(summary)} markets ({args.strategy})")
    print("\naggregate:")
    print(summary[["total_pnl", "sharpe_annualized", "hit_rate", "max_drawdown", "exposure"]].describe())
    print("\ntop 10 by total PnL:")
    print(summary.head(10))


if __name__ == "__main__":
    main()
