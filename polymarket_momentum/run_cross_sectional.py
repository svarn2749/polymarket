from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cross_sectional import (
    backtest_cross_sectional,
    load_panel,
    sweep_cross_sectional,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cross-sectional momentum/reversion backtest."
    )
    parser.add_argument("--data", type=Path, default=Path("data/prices"))
    parser.add_argument("--lookback", type=int, default=24, help="lookback in hourly bars")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--strategy",
        choices=["momentum", "reversion"],
        default="reversion",
    )
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=200.0)
    parser.add_argument("--resample", type=str, default="1h")
    parser.add_argument("--out", type=Path, default=Path("data/cross_sectional.csv"))
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the default sweep grid and print a summary table.",
    )
    args = parser.parse_args()

    panel = load_panel(args.data, resample=args.resample)
    if panel.empty:
        print(f"no usable data in {args.data}")
        return

    n_bars, n_markets = panel.shape
    density = panel.notna().to_numpy().mean()
    print(f"panel: {n_bars} bars x {n_markets} markets (density={density:.2%})")

    result = backtest_cross_sectional(
        panel,
        lookback_hours=args.lookback,
        top_k=args.top_k,
        strategy=args.strategy,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print(f"\nprimary config: {args.strategy} lookback={args.lookback} top_k={args.top_k} "
          f"slippage_bps={args.slippage_bps}")
    for key, value in result.stats.items():
        if isinstance(value, float):
            print(f"  {key:24s} {value:,.4f}")
        else:
            print(f"  {key:24s} {value}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.bars.to_csv(args.out)
    print(f"\nper-bar PnL + equity written to {args.out}")

    if args.sweep:
        print("\nsweep grid (lookback x top_k x strategy, sorted by Sharpe):")
        grid = sweep_cross_sectional(
            panel,
            lookbacks=[12, 24, 72],
            top_ks=[5, 10, 25],
            strategies=["momentum", "reversion"],
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )
        print(grid.to_string(index=False))
        sweep_path = args.out.with_name(args.out.stem + "_sweep.csv")
        grid.to_csv(sweep_path, index=False)
        print(f"\nsweep written to {sweep_path}")


if __name__ == "__main__":
    main()
