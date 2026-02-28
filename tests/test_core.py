import numpy as np
import pandas as pd
import pytest

from backtester import engine, strategies
from pricing.black_scholes import bs_price, implied_vol, mc_price


def test_put_call_parity():
    S, K, T, r, sigma = 100, 95, 1.0, 0.03, 0.3
    call, put = bs_price(S, K, T, r, sigma, True), bs_price(S, K, T, r, sigma, False)
    assert call - put == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)


def test_known_bs_value():
    # Hull, Options Futures & Other Derivatives, Example 15.6
    assert bs_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.76, abs=0.01)


@pytest.mark.parametrize("call", [True, False])
def test_mc_matches_closed_form(call):
    args = (100, 110, 0.75, 0.05, 0.2)
    res = mc_price(*args, call=call, n_paths=200_000, seed=3)
    assert abs(res.price - bs_price(*args, call=call)) < 4 * res.std_error


def test_control_variate_reduces_error():
    args = (100, 90, 1.0, 0.02, 0.25)
    plain = mc_price(*args, seed=5, antithetic=False, control_variate=False)
    reduced = mc_price(*args, seed=5)
    assert reduced.std_error < plain.std_error / 2


def test_implied_vol_roundtrip():
    price = bs_price(100, 120, 0.25, 0.01, 0.42)
    assert implied_vol(price, 100, 120, 0.25, 0.01) == pytest.approx(0.42, abs=1e-6)


def test_no_lookahead():
    # A position set from bar t's return must only earn bar t+1's return.
    # If the engine forgot the lag, sign(r_t) * r_t would give a huge Sharpe.
    prices = engine.synthetic_prices(n=2000, seed=1)
    same_bar = lambda p: np.sign(p["close"].pct_change()).fillna(0)
    res = engine.run(prices, same_bar, cost_bps=0)
    assert abs(res.metrics["sharpe"]) < 1.5


def test_buy_and_hold_tracks_asset():
    prices = engine.synthetic_prices(n=300, seed=2)
    res = engine.run(prices, strategies.buy_and_hold, cost_bps=0)
    asset_total = prices["close"].iloc[-1] / prices["close"].iloc[0] - 1
    assert res.metrics["total_return"] == pytest.approx(asset_total, rel=1e-9)


def test_costs_reduce_returns():
    prices = engine.synthetic_prices(n=1000, seed=4)
    strat = strategies.zscore_mean_reversion(10)
    free = engine.run(prices, strat, cost_bps=0).metrics["total_return"]
    costly = engine.run(prices, strat, cost_bps=10).metrics["total_return"]
    assert costly < free
