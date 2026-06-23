"""Offline tests for profit mode (Monte-Carlo over random months)."""

from __future__ import annotations

import pandas as pd

from quant.profit import (
    format_profit,
    monthly_returns,
    run_profit,
    simulate_profit,
)


def test_monthly_returns_are_aligned_monthly_series(regime_switch_prices):
    strat_m, bh_m = monthly_returns(regime_switch_prices)
    assert isinstance(strat_m.index, pd.PeriodIndex)
    assert list(strat_m.index) == list(bh_m.index)
    assert len(strat_m) > 1


def test_uptrend_is_always_profitable(uptrend_prices):
    strat_m, bh_m = monthly_returns(uptrend_prices)
    s_stats, _b_stats, sampling, example = simulate_profit(
        strat_m, bh_m, months=6, amount=1000.0, trials=200, seed=7
    )
    # A persistent uptrend is always BULL -> long -> every month positive.
    assert s_stats.prob_profit == 1.0
    assert s_stats.median > 1000.0
    assert sampling == "without replacement"
    assert len(example) == 6


def test_bootstrap_when_too_few_months(uptrend_prices):
    strat_m, bh_m = monthly_returns(uptrend_prices)
    # Far more months requested than exist -> sample with replacement.
    _s, _b, sampling, example = simulate_profit(
        strat_m, bh_m, months=500, amount=1000.0, trials=10, seed=1
    )
    assert "bootstrap" in sampling
    assert len(example) == 500


def test_seed_is_reproducible(noisy_prices):
    strat_m, bh_m = monthly_returns(noisy_prices)
    a = simulate_profit(strat_m, bh_m, months=8, trials=100, seed=123)[0]
    b = simulate_profit(strat_m, bh_m, months=8, trials=100, seed=123)[0]
    assert a == b


def test_run_profit_end_to_end_and_format(regime_switch_prices):
    result = run_profit(
        prices=regime_switch_prices, months=6, amount=2500.0, trials=100, seed=0
    )
    assert result.amount == 2500.0
    assert result.months == 6
    assert result.num_available_months > 0
    text = format_profit(result)
    assert "PROFIT MODE" in text
    assert "$2,500" in text
