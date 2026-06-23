"""Tests for regime state labeling (elements 1 & 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.states import (
    State,
    label_states,
    trailing_return,
)


def test_uptrend_is_bull(uptrend_prices):
    labels = label_states(uptrend_prices, window=20)
    # A steady +0.4%/day climb sums to ~8% over 20 days -> all BULL.
    assert (labels == State.BULL).all()


def test_downtrend_is_bear(downtrend_prices):
    labels = label_states(downtrend_prices, window=20)
    assert (labels == State.BEAR).all()


def test_thresholds_are_applied():
    # Construct a series whose 20-day trailing return sits in the sideways band.
    rets = np.full(60, 0.001)  # ~2% over 20 days -> between -5% and +5%
    prices = pd.Series(100 * np.cumprod(1 + np.insert(rets, 0, 0)))
    labels = label_states(prices, window=20)
    assert (labels == State.SIDEWAYS).all()


def test_labels_drop_lookback_window(noisy_prices):
    window = 20
    labels = label_states(noisy_prices, window=window)
    # First labelable day is index `window` (need a full look-back of returns).
    assert labels.index[0] == noisy_prices.index[window]
    assert len(labels) == len(noisy_prices) - window


def test_invalid_thresholds_raise():
    prices = pd.Series(100 * np.ones(50))
    with pytest.raises(ValueError):
        label_states(prices, bull_threshold=0.0, bear_threshold=0.0)


def test_jump_penalty_zero_matches_greedy(noisy_prices):
    """jump_penalty=0 must reproduce the greedy threshold labeling exactly."""
    greedy = label_states(noisy_prices, window=20, jump_penalty=0.0)
    explicit_zero = label_states(noisy_prices, window=20)
    pd.testing.assert_series_equal(greedy, explicit_zero)


def test_jump_penalty_reduces_switches(noisy_prices):
    """A positive switch penalty must yield a more persistent (fewer-switch) path."""
    greedy = label_states(noisy_prices, window=20, jump_penalty=0.0)
    penalized = label_states(noisy_prices, window=20, jump_penalty=0.5)
    switches_greedy = int((greedy.diff() != 0).sum())
    switches_pen = int((penalized.diff() != 0).sum())
    assert switches_pen < switches_greedy
    assert len(penalized) == len(greedy)


def test_jump_penalty_large_collapses_to_single_regime(noisy_prices):
    """A very large penalty makes any switch uneconomic -> one regime throughout."""
    labels = label_states(noisy_prices, window=20, jump_penalty=1e6)
    assert labels.nunique() == 1


def test_negative_jump_penalty_raises(noisy_prices):
    with pytest.raises(ValueError):
        label_states(noisy_prices, window=20, jump_penalty=-0.1)


def test_trailing_return_matches_manual_sum(noisy_prices):
    window = 20
    tr = trailing_return(noisy_prices, window)
    manual = noisy_prices.pct_change().rolling(window).sum()
    pd.testing.assert_series_equal(tr, manual, check_names=False)
