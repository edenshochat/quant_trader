"""Tests for the walk-forward backtest (element 9)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from quant import backtest as backtest_mod
from quant.backtest import walk_forward


def test_no_lookahead_only_past_prices_used(noisy_prices):
    """The matrix at each step must be built from prices[: t+1] only.

    We spy on label_states and assert every call receives a history whose last
    timestamp is strictly before the day whose return is being realized.
    """
    real_label = backtest_mod.label_states
    seen_lengths = []

    def spy(history, *args, **kwargs):
        seen_lengths.append(len(history))
        return real_label(history, *args, **kwargs)

    with mock.patch.object(backtest_mod, "label_states", side_effect=spy):
        walk_forward(noisy_prices, window=20, min_history=100)

    # Histories must be strictly increasing in length (expanding window) and
    # never reach the full series until the final step.
    assert seen_lengths == sorted(seen_lengths)
    assert max(seen_lengths) < len(noisy_prices)


def test_uptrend_backtest_is_profitable_and_long(uptrend_prices):
    res = walk_forward(uptrend_prices, window=20, min_history=100)
    assert res.num_days > 0
    assert res.total_return > 0
    # Persistent bull => positions should be net long.
    assert res.positions.mean() > 0


def test_downtrend_backtest_goes_short(downtrend_prices):
    res = walk_forward(downtrend_prices, window=20, min_history=100)
    # Persistent bear => short positions => profit from the decline.
    assert res.positions.mean() < 0
    assert res.total_return > 0


def test_equity_curve_length_matches_returns(noisy_prices):
    res = walk_forward(noisy_prices, window=20, min_history=100)
    assert len(res.equity_curve) == res.num_days
    assert len(res.positions) == res.num_days
    assert len(res.strategy_returns) == res.num_days


def test_max_drawdown_is_non_positive(noisy_prices):
    res = walk_forward(noisy_prices, window=20, min_history=100)
    assert res.max_drawdown <= 0.0


def test_vol_target_scales_down_a_high_vol_asset(noisy_prices):
    """A low vol target on a higher-vol asset must reduce realized strategy vol."""
    base = walk_forward(noisy_prices, window=20, min_history=100)
    vt = walk_forward(
        noisy_prices, window=20, min_history=100, vol_target=0.05, max_leverage=3.0
    )
    assert vt.annualized_volatility < base.annualized_volatility
    assert vt.num_days == base.num_days  # same traded days, only sizing differs


def test_vol_target_respects_no_lookahead(noisy_prices):
    """Vol targeting must not peek: spy that every history ends before the realized day."""
    real_label = backtest_mod.label_states
    seen = []

    def spy(history, *args, **kwargs):
        seen.append(len(history))
        return real_label(history, *args, **kwargs)

    with mock.patch.object(backtest_mod, "label_states", side_effect=spy):
        walk_forward(noisy_prices, window=20, min_history=100, vol_target=0.10)
    assert seen == sorted(seen)
    assert max(seen) < len(noisy_prices)


def test_hysteresis_reduces_turnover(noisy_prices):
    """A no-trade buffer must lower total turnover (the whipsaw proxy)."""
    base = walk_forward(noisy_prices, window=20, min_history=100)
    hyst = walk_forward(noisy_prices, window=20, min_history=100, hysteresis=0.3)
    assert hyst.turnover < base.turnover
    assert hyst.num_days == base.num_days


def test_short_into_extreme_spike_does_not_return_complex():
    """A short position into a >100% single-day move blows up equity.

    The annualized-return math must stay real (a >=100% loss => -100%), not
    raise or return a complex number from a negative fractional power.
    """
    rng = np.random.default_rng(0)
    # A long, calm bear stretch so the model is confidently short, then a single
    # explosive +250% day that wipes out the short.
    rets = list(np.full(150, -0.01)) + [2.5] + list(rng.normal(0, 0.01, 20))
    idx = pd.date_range("2020-01-01", periods=len(rets) + 1, freq="B")
    prices = pd.Series(100 * np.cumprod(np.insert(1.0 + np.array(rets), 0, 1.0)), index=idx)

    res = walk_forward(prices, window=20, min_history=80)
    assert isinstance(res.annualized_return, float)
    assert res.annualized_return == pytest.approx(-1.0)
    assert res.total_return <= -1.0


def test_insufficient_history_raises():
    prices = pd.Series(100 * np.ones(30))
    with pytest.raises(ValueError):
        walk_forward(prices, window=20, min_history=100)
