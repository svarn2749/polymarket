from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd

from .data import get_orderbook


def snapshot_spreads(
    markets_csv: Path,
    out_csv: Path,
    *,
    concurrency: int = 8,
) -> pd.DataFrame:
    markets = pd.read_csv(markets_csv, dtype={"id": str, "yes_token_id": str, "no_token_id": str})
    limits = httpx.Limits(
        max_keepalive_connections=concurrency, max_connections=concurrency * 2
    )
    rows: list[dict] = []
    skipped = 0
    errors = 0

    with httpx.Client(timeout=15, limits=limits) as client:
        def fetch_one(row: dict) -> tuple[dict, dict | None, bool]:
            try:
                book = get_orderbook(row["yes_token_id"], client=client)
                return row, book, False
            except Exception:
                return row, None, True

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(fetch_one, r) for r in markets.to_dict("records")]
            done = 0
            for fut in as_completed(futures):
                row, book, err = fut.result()
                done += 1
                if err:
                    errors += 1
                    continue
                if book is None:
                    skipped += 1
                    continue
                rows.append(
                    {
                        "market_id": row["id"],
                        "bid": book["bid"],
                        "ask": book["ask"],
                        "mid": book["mid"],
                        "spread_bps": book["spread_bps"],
                    }
                )
                if done % 25 == 0:
                    print(f"  [{done}/{len(markets)}] ok={len(rows)} skipped={skipped} errors={errors}")

    df = pd.DataFrame(rows, columns=["market_id", "bid", "ask", "mid", "spread_bps"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"wrote {out_csv}: {len(df)} tradeable / {skipped} un-tradeable / {errors} errors"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot live Polymarket orderbook spreads per market."
    )
    parser.add_argument("--markets", type=Path, default=Path("data/markets.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/spreads.csv"))
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    df = snapshot_spreads(args.markets, args.out, concurrency=args.concurrency)
    if df.empty:
        print("no spreads captured")
        return
    q = df["spread_bps"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    print("spread_bps quantiles:")
    for k, v in q.items():
        print(f"  p{int(k * 100):>2}: {v:8.1f} bps")


if __name__ == "__main__":
    main()
