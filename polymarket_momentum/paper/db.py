from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    side TEXT NOT NULL,          -- 'buy' or 'sell'
    size REAL NOT NULL,          -- signed: +buy, -sell (shares of YES token)
    price REAL NOT NULL,
    signal REAL,
    strategy TEXT,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id, ts);

CREATE TABLE IF NOT EXISTS signals (
    ts INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    value REAL,
    direction REAL,              -- desired position direction (-1, 0, +1)
    PRIMARY KEY (ts, market_id)
);

CREATE TABLE IF NOT EXISTS snapshots (
    ts INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    bid REAL,
    ask REAL,
    mid REAL,
    PRIMARY KEY (ts, market_id)
);

CREATE TABLE IF NOT EXISTS equity (
    ts INTEGER PRIMARY KEY,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    total_pnl REAL NOT NULL,
    gross_exposure REAL NOT NULL,
    n_positions INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS poll_log (
    ts INTEGER PRIMARY KEY,
    duration_ms INTEGER,
    n_markets INTEGER,
    n_errors INTEGER,
    n_trades INTEGER,
    note TEXT
);
"""


def init(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
