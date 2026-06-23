"""Tests for the Sharpe-significance guardrails."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def test_psr_high_for_strong_consistent_returns():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.005, 1000))  # high, steady Sharpe
    assert probabilistic_sharpe_ratio(r) > 0.99


def test_psr_near_half_for_zero_mean_noise():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.01, 2000))
    # Zero-mean noise => no strong evidence of a positive Sharpe either way.
    assert 0.15 < probabilistic_sharpe_ratio(r) < 0.85


def test_psr_in_unit_interval():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0003, 0.012, 500))
    psr = probabilistic_sharpe_ratio(r)
    assert 0.0 <= psr <= 1.0


def test_expected_max_sharpe_grows_with_trials():
    a = expected_max_sharpe(sr_trials_std=0.05, n_trials=10)
    b = expected_max_sharpe(sr_trials_std=0.05, n_trials=1000)
    assert b > a > 0


def test_deflated_is_not_greater_than_plain_psr():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0008, 0.01, 1500))
    plain = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    deflated = deflated_sharpe_ratio(r, n_trials=50, sr_trials_std=0.05)
    # Deflating against the expected max of many trials can only lower confidence.
    assert deflated <= plain
