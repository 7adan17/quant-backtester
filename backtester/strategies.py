from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(prices: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=prices.index)


def sma_crossover(fast: int = 20, slow: int = 100):
    def strategy(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        signal = np.sign(close.rolling(fast).mean() - close.rolling(slow).mean())
        return signal.fillna(0)
    strategy.__name__ = f"sma_{fast}_{slow}"
    return strategy


def zscore_mean_reversion(window: int = 20, entry: float = 1.5, exit_: float = 0.5):
    """Short when price is stretched above its rolling mean, long when below,
    flat once the z-score reverts inside the exit band."""
    def strategy(prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        z = (close - close.rolling(window).mean()) / close.rolling(window).std()
        pos = np.zeros(len(z))
        for i, zi in enumerate(z.to_numpy()):
            prev = pos[i - 1] if i else 0.0
            if np.isnan(zi):
                pos[i] = 0.0
            elif zi > entry:
                pos[i] = -1.0
            elif zi < -entry:
                pos[i] = 1.0
            elif abs(zi) < exit_:
                pos[i] = 0.0
            else:
                pos[i] = prev
        return pd.Series(pos, index=prices.index)
    strategy.__name__ = f"zscore_{window}"
    return strategy


def vol_target(base, target_vol: float = 0.10, lookback: int = 20):
    """Scales another strategy's positions so realized volatility stays near target_vol."""
    def strategy(prices: pd.DataFrame) -> pd.Series:
        realized = prices["close"].pct_change().rolling(lookback).std() * np.sqrt(252)
        scale = (target_vol / realized).clip(upper=1.0).fillna(0)
        return base(prices) * scale
    strategy.__name__ = f"voltgt_{base.__name__}"
    return strategy
