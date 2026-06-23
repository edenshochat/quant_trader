"""Deterministic synthetic price fixtures — no network required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _series(returns: np.ndarray, start: float = 100.0) -> pd.Series:
    """Build a price series from a daily-return array with a date index."""
    prices = start * np.cumprod(1.0 + returns)
    prices = np.insert(prices, 0, start)
    idx = pd.date_range("2015-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=idx, name="SYNTH")


@pytest.fixture
def uptrend_prices() -> pd.Series:
    """Steady ~+0.4%/day climb — should be persistently BULL."""
    return _series(np.full(400, 0.004))


@pytest.fixture
def downtrend_prices() -> pd.Series:
    """Steady ~-0.4%/day decline — should be persistently BEAR."""
    return _series(np.full(400, -0.004))


@pytest.fixture
def regime_switch_prices() -> pd.Series:
    """Long bull block followed by a long bear block.

    Gives the transition matrix and the HMM clearly separated regimes to learn.
    """
    rng = np.random.default_rng(0)
    bull = 0.006 + rng.normal(0, 0.002, 300)
    bear = -0.006 + rng.normal(0, 0.002, 300)
    return _series(np.concatenate([bull, bear]))


@pytest.fixture
def noisy_prices() -> pd.Series:
    """Random walk with tiny drift — a realistic mix of all three states."""
    rng = np.random.default_rng(42)
    return _series(rng.normal(0.0003, 0.012, 800))
