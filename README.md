# polymarket-momentum

Research harness for systematic betting strategies on event markets. Supports
Polymarket and (read-only) Kalshi.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Pipeline

1. **Fetch** markets + price history into `data/`:
   ```bash
   python -m polymarket_momentum.fetch --source polymarket --limit 500 --lookback-days 120
   python -m polymarket_momentum.fetch --source kalshi   --limit 500 --lookback-days 30
   ```
   For Polymarket, `--include-closed` (default on) also pulls resolved markets;
   the last `--resolution-trim-hours` (default 24h) are dropped to avoid the
   resolution jump polluting the signal.

2. **Backtest**:
   ```bash
   python -m polymarket_momentum.run_backtest --lookback 24 --threshold 0.05
   ```
   `--fee-bps` adds a flat per-turnover cost. `--slippage-bps` adds a
   round-trip slippage assumption; `--spreads-csv` overrides it per-market
   from a `market_id,spread_bps` file.

3. **Dashboard** (most-attractive bets + current positions):
   ```bash
   export POLYMARKET_WALLET=0x...
   python -m polymarket_momentum.serve --reload
   ```
   Open `http://127.0.0.1:8000/`. Toggle source via the selector or
   `?source=kalshi`. JSON: `/api/ranking`, `/api/positions`.

## Layout

- `sources/` — venue abstraction. `base.py` defines `Market`, `OrderBook`,
  `Position`, `FeeModel` (`FlatFee`, `KalshiFee`), and the `MarketSource`
  protocol. `polymarket.py` and `kalshi.py` implement it. `get_source(name)`
  is the registry entry point.
- `strategy.py` — momentum signal + position sizing.
- `backtest.py` — per-market event-loop backtest. Costs are `fee_bps + slippage_bps`
  of notional, with optional per-market spread override.
- `cost_model.py` — orderbook walk for a target notional, plus pluggable
  `FeeModel`. Shared by `ranking.py` and any live-cost calculation.
- `ranking.py` — scores markets by `|signal × price| − round-trip cost`,
  ranks top-N. Parallel orderbook fetch.
- `web/` — FastAPI dashboard (Jinja2 templates, no build step).
- `data.py` — legacy Polymarket-only module retained for the research
  scripts (`cross_sectional.py`, `oos.py`, `sweep.py`, `snapshot_spreads.py`).
  New code should use `sources/`.

## Strategy

At each hourly bar: trailing-`lookback` return on YES. If `|return| > threshold`,
hold `sign(return)` until the next bar; otherwise flat. PnL is the signed price
change minus cost. Mean-reversion is the same logic with an inverted sign.

## Limitations

- Orderbook-aware costs are live-only; backtest uses a flat bps (or per-market
  from `data/spreads.csv` via `snapshot_spreads`).
- Kalshi positions require authenticated API access (not wired).
- Polymarket positions need `POLYMARKET_WALLET` set.
- No live order placement.
