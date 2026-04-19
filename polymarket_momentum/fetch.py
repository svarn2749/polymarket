from __future__ import annotations

import argparse
from pathlib import Path

from .sources import get_source
from .sources.polymarket import PolymarketSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Download market price history.")
    parser.add_argument(
        "--source",
        choices=["polymarket", "kalshi"],
        default="polymarket",
    )
    parser.add_argument("--out", type=Path, default=Path("data/prices"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--min-volume", type=float, default=50_000)
    parser.add_argument("--fidelity", type=int, default=60, help="minutes between samples")
    parser.add_argument(
        "--interval",
        default="1m",
        help="Polymarket preset window (used if --lookback-days is not set): 1h, 6h, 1d, 1w, 1m, max",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="If set, chunk-fetch this many days of history (overrides --interval).",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--include-closed",
        dest="include_closed",
        action="store_true",
        default=True,
        help="Also list closed markets (polymarket only). Default: on.",
    )
    parser.add_argument(
        "--no-include-closed",
        dest="include_closed",
        action="store_false",
    )
    parser.add_argument(
        "--resolution-trim-hours",
        type=float,
        default=24.0,
        help="Drop the last N hours for closed markets (avoids resolution jump).",
    )
    args = parser.parse_args()

    source = get_source(args.source)

    if isinstance(source, PolymarketSource):
        markets = source.list_markets(
            min_volume=args.min_volume,
            limit=args.limit,
            include_closed=args.include_closed,
        )
        print(
            f"[polymarket] listed {len(markets)} markets "
            f"(min_volume={args.min_volume:.0f} include_closed={args.include_closed})"
        )
        source.write_market_metadata(markets, args.out.parent / "markets.csv")
        written = source.cache_prices(
            markets,
            args.out,
            fidelity=args.fidelity,
            interval=args.interval,
            lookback_days=args.lookback_days,
            concurrency=args.concurrency,
            resolution_trim_hours=args.resolution_trim_hours,
        )
        print(f"cached prices for {len(written)} markets -> {args.out}")
        return

    markets = source.list_markets(min_volume=args.min_volume, limit=args.limit)
    print(f"[{args.source}] listed {len(markets)} markets (min_volume={args.min_volume:.0f})")
    source.write_market_metadata(markets, args.out.parent / "markets.csv")
    written = source.cache_prices(
        markets,
        args.out,
        fidelity=args.fidelity,
        lookback_days=args.lookback_days,
        concurrency=args.concurrency,
    )
    print(f"cached prices for {len(written)} markets -> {args.out}")


if __name__ == "__main__":
    main()
