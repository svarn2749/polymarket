from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_signal(prices: pd.Series, lookback_bars: int) -> pd.Series:
    return prices / prices.shift(lookback_bars) - 1.0


def sized_position(
    signal: pd.Series,
    *,
    entry_threshold: float,
    max_position: float = 1.0,
) -> pd.Series:
    raw = np.where(signal.abs() > entry_threshold, np.sign(signal), 0.0)
    return pd.Series(raw, index=signal.index).clip(-max_position, max_position)
