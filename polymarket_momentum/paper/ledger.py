from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class PaperPosition:
    market_id: str
    size: float            # signed: + long YES, - short YES
    avg_entry_price: float
    realized_pnl: float    # closed-lot PnL so far
    last_price: float | None = None

    @property
    def unrealized_pnl(self) -> float:
        if self.last_price is None or self.size == 0:
            return 0.0
        return (self.last_price - self.avg_entry_price) * self.size

    @property
    def notional(self) -> float:
        if self.last_price is None:
            return abs(self.size) * self.avg_entry_price
        return abs(self.size) * self.last_price


def record_trade(
    conn: sqlite3.Connection,
    *,
    ts: int,
    market_id: str,
    side: str,
    size: float,            # signed
    price: float,
    signal: float | None,
    strategy: str,
    source: str,
) -> None:
    conn.execute(
        "INSERT INTO trades (ts, market_id, side, size, price, signal, strategy, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, market_id, side, size, price, signal, strategy, source),
    )


def current_positions(
    conn: sqlite3.Connection,
    *,
    strategy: str | None = None,
) -> dict[str, PaperPosition]:
    """Reconstruct current positions from the full trades log.

    Uses weighted-average entry price; realized PnL accumulates on closes /
    reversals. Rebuilt from scratch each time for simplicity — trades log is
    the source of truth. Pass `strategy` to scope to one strategy's trades.
    """
    if strategy is None:
        rows = conn.execute(
            "SELECT ts, market_id, size, price FROM trades ORDER BY id ASC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ts, market_id, size, price FROM trades "
            "WHERE strategy = ? ORDER BY id ASC",
            (strategy,),
        ).fetchall()

    state: dict[str, PaperPosition] = {}
    for row in rows:
        mid = row["market_id"]
        size = float(row["size"])
        price = float(row["price"])
        pos = state.get(mid) or PaperPosition(market_id=mid, size=0.0, avg_entry_price=0.0, realized_pnl=0.0)

        old_size = pos.size
        new_size = old_size + size
        # Same side (adding): weighted-average entry.
        if old_size == 0 or (old_size > 0 and size > 0) or (old_size < 0 and size < 0):
            total_cost = old_size * pos.avg_entry_price + size * price
            pos.avg_entry_price = total_cost / new_size if new_size != 0 else 0.0
            pos.size = new_size
        else:
            # Opposite side (reducing or flipping).
            close_qty = min(abs(size), abs(old_size))
            direction = 1 if old_size > 0 else -1
            pos.realized_pnl += direction * close_qty * (price - pos.avg_entry_price)
            pos.size = new_size
            if (old_size > 0 and new_size < 0) or (old_size < 0 and new_size > 0):
                # Flipped through zero — new leg opens at `price`.
                pos.avg_entry_price = price
            elif new_size == 0:
                pos.avg_entry_price = 0.0
            # else: reduced but same side — avg_entry_price unchanged.

        state[mid] = pos

    # Keep closed positions that still carry realized PnL; the dashboard
    # can filter to open (size != 0) positions when displaying.
    return {
        k: v for k, v in state.items()
        if abs(v.size) > 1e-9 or abs(v.realized_pnl) > 1e-9
    }


def record_equity(
    conn: sqlite3.Connection,
    *,
    ts: int,
    strategy: str,
    realized: float,
    unrealized: float,
    gross_exposure: float,
    n_positions: int,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO equity "
        "(ts, strategy, realized_pnl, unrealized_pnl, total_pnl, gross_exposure, n_positions) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, strategy, realized, unrealized, realized + unrealized, gross_exposure, n_positions),
    )


def record_poll(
    conn: sqlite3.Connection,
    *,
    ts: int,
    duration_ms: int,
    n_markets: int,
    n_errors: int,
    n_trades: int,
    note: str = "",
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO poll_log "
        "(ts, duration_ms, n_markets, n_errors, n_trades, note) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, duration_ms, n_markets, n_errors, n_trades, note),
    )
