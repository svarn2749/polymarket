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


def sized_position_hysteresis(
    signal: pd.Series,
    *,
    entry_threshold: float,
    exit_threshold: float,
    max_position: float = 1.0,
) -> pd.Series:
    """Stateful ±1 position with enter/exit bands.

    - Enter long  when signal >  entry_threshold
    - Enter short when signal < -entry_threshold
    - Exit when |signal| < exit_threshold
    - Flip directly if signal crosses from < -entry to > +entry (or vice versa)
    - NaN signals preserve the current position (no forced exit)

    exit_threshold should be <= entry_threshold; otherwise this degrades to
    the stateless behavior.
    """
    values = signal.to_numpy()
    out = np.zeros(len(values), dtype=float)
    current = 0.0
    for i, s in enumerate(values):
        if np.isnan(s):
            out[i] = current
            continue
        if current == 0.0:
            if s > entry_threshold:
                current = 1.0
            elif s < -entry_threshold:
                current = -1.0
        else:
            if current > 0 and s < -entry_threshold:
                current = -1.0
            elif current < 0 and s > entry_threshold:
                current = 1.0
            elif abs(s) < exit_threshold:
                current = 0.0
        out[i] = current
    return pd.Series(out, index=signal.index).clip(-max_position, max_position)
