from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cross_sectional import load_panel, sweep_cross_sectional


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Out-of-sample validation for cross-sectional sweep (time-split panel)."
    )
    parser.add_argument("--data", type=Path, default=Path("data/prices"))
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=200.0)
    parser.add_argument("--resample", type=str, default="1h")
    parser.add_argument("--out", type=Path, default=Path("data/oos_cross_sectional.csv"))
    args = parser.parse_args()

    panel = load_panel(args.data, resample=args.resample)
    if panel.empty:
        print(f"no usable data in {args.data}")
        return

    n_bars = len(panel)
    cut = int(n_bars * args.train_frac)
    train_panel = panel.iloc[:cut]
    test_panel = panel.iloc[cut:]
    print(
        f"panel: {n_bars} bars x {panel.shape[1]} markets; "
        f"train={len(train_panel)} bars, test={len(test_panel)} bars"
    )

    grid_kwargs = dict(
        lookbacks=[12, 24, 72],
        top_ks=[5, 10, 25],
        strategies=["momentum", "reversion"],
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    print("\nsweeping on train half...")
    train = sweep_cross_sectional(train_panel, **grid_kwargs)
    print("sweeping on test half...")
    test = sweep_cross_sectional(test_panel, **grid_kwargs)

    keys = ["strategy", "lookback", "top_k"]
    merged = train.merge(test, on=keys, suffixes=("_train", "_test"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    pearson = merged[["sharpe_train", "sharpe_test"]].corr().iloc[0, 1]
    spearman = merged[["sharpe_train", "sharpe_test"]].corr(method="spearman").iloc[0, 1]
    print(
        f"\ntrain/test Sharpe correlation: pearson={pearson:.3f} spearman={spearman:.3f}"
    )
    print("  >0.5 = signal probably real, 0-0.5 = weak, <0 = overfit")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    cols = [
        "strategy", "lookback", "top_k",
        "sharpe_train", "sharpe_test",
        "total_pnl_train", "total_pnl_test",
        "frac_positive_bars_train", "frac_positive_bars_test",
    ]
    print("\ntop 5 train configs — how they hold up on test:")
    print(merged.sort_values("sharpe_train", ascending=False).head(5)[cols].to_string(index=False))

    print("\ntop 5 test configs:")
    print(merged.sort_values("sharpe_test", ascending=False).head(5)[cols].to_string(index=False))

    best_train = merged.sort_values("sharpe_train", ascending=False).iloc[0]
    print(
        f"\nbest-on-train: {best_train['strategy']}, "
        f"lookback={best_train['lookback']}, top_k={best_train['top_k']} "
        f"-> train Sharpe {best_train['sharpe_train']:.2f}, test Sharpe {best_train['sharpe_test']:.2f}"
    )


if __name__ == "__main__":
    main()
