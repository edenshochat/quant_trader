"""Tests for cross-asset / market-regime backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest import walk_forward
from quant.cross_asset import build_market_index, own_vs_market


def _series(returns: np.ndarray, start: float = 100.0, seed_idx: int = 0) -> pd.Series:
    prices = start * np.cumprod(1.0 + np.insert(returns, 0, 0.0))
    idx = pd.date_range("2016-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=idx)


def test_market_index_is_equalweight_rebased():
    a = _series(np.full(300, 0.001))
    b = _series(np.full(300, -0.001))
    idx = build_market_index({"A": a, "B": b})
    assert idx.iloc[0] == pytest.approx(1.0)
    assert idx.name == "MARKET"
    assert len(idx) == 301


def test_empty_basket_raises():
    with pytest.raises(ValueError):
        build_market_index({})


def test_signal_prices_default_equals_self(noisy_prices):
    """signal_prices=None must equal trading the asset off its own series."""
    a = walk_forward(noisy_prices, window=20, min_history=100)
    b = walk_forward(noisy_prices, window=20, min_history=100, signal_prices=noisy_prices)
    assert a.sharpe == pytest.approx(b.sharpe)
    assert a.num_days == b.num_days


def test_market_regime_trades_target_returns(uptrend_prices, downtrend_prices):
    """Returns must come from the target, regime from the signal series.

    Trade a downtrending target off an uptrending market regime: the market is
    persistently BULL -> long signal -> applied to a falling asset -> losses.
    """
    own, market = own_vs_market(
        downtrend_prices, uptrend_prices, window=20, min_history=100
    )
    # Off its own (bear) regime the model shorts the decline and profits;
    # off the (bull) market regime it goes long the decline and loses.
    assert own.total_return > 0
    assert market.total_return < own.total_return


def test_beta_align_flips_a_perfect_hedge(uptrend_prices, downtrend_prices):
    """A target perfectly anti-correlated to the market should be sign-flipped.

    The market is BULL (long signal); the target falls. Without alignment the
    long loses; beta_align detects the negative correlation, shorts the fall,
    and turns the loss into a gain.
    """
    _own, plain = own_vs_market(downtrend_prices, uptrend_prices, window=20, min_history=100)
    _own2, aligned = own_vs_market(
        downtrend_prices, uptrend_prices, beta_align=True, window=20, min_history=100
    )
    assert plain.total_return < 0  # long a falling asset off the bull regime
    assert aligned.total_return > plain.total_return  # flipped -> profits from the fall
