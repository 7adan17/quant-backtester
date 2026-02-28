"""Vectorized daily backtester.

A strategy maps a price DataFrame to target positions in [-1, 1]. The engine
lags positions by one bar (no look-ahead), applies transaction costs on
turnover, and reports standard performance metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252

Strategy = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    metrics: dict

    def summary(self) -> str:
        return "\n".join(f"{k:>16}: {v:,.4f}" for k, v in self.metrics.items())


def load_csv(path: str, date_col: str = "Date", close_col: str = "Close") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    return df.rename(columns={close_col: "close"})[["close"]]


def synthetic_prices(n: int = 2520, mu: float = 0.07, sigma: float = 0.2,
                     s0: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Geometric Brownian motion sample path, for testing without market data."""
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"close": s0 * np.exp(np.cumsum(log_ret))}, index=idx)


def compute_metrics(returns: pd.Series, positions: pd.Series) -> dict:
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "ann_vol": vol,
        "sharpe": returns.mean() * TRADING_DAYS / vol if vol > 0 else np.nan,
        "sortino": returns.mean() * TRADING_DAYS / downside if downside > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "hit_rate": (returns[positions.shift(1).fillna(0) != 0] > 0).mean(),
        "turnover": positions.diff().abs().sum() / years if years > 0 else np.nan,
    }


def run(prices: pd.DataFrame, strategy: Strategy, cost_bps: float = 1.0) -> BacktestResult:
    target = strategy(prices).reindex(prices.index).fillna(0).clip(-1, 1)
    asset_ret = prices["close"].pct_change().fillna(0)

    # Position decided at close t earns the return from t to t+1.
    held = target.shift(1).fillna(0)
    costs = target.diff().abs().fillna(target.abs()) * cost_bps / 1e4
    strat_ret = held * asset_ret - costs.shift(1).fillna(0)

    return BacktestResult(
        equity=(1 + strat_ret).cumprod(),
        returns=strat_ret,
        positions=target,
        metrics=compute_metrics(strat_ret, target),
    )
