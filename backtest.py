"""Element 9: walk-forward backtest (no look-ahead).

A naive backtest fits the transition matrix on the *whole* history and then
"replays" the past with it — but that matrix already knows the future, so the
result is fiction. Walk-forward fixes this: on each day ``t`` we rebuild the
states and the transition matrix using only data up to and including ``t``,
read today's state, compute the position, and then realize that position
against the *next* day's return. The matrix is recomputed every step, so no
information from ``t+1`` onward ever leaks into the decision made at ``t``.

This is computationally heavy by hand (the whole transcript's caveat) but cheap
in NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import position_size
from .states import (
    DEFAULT_BEAR_THRESHOLD,
    DEFAULT_BULL_THRESHOLD,
    DEFAULT_WINDOW,
    label_states,
)
from .transition import transition_matrix

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    """Summary stats plus the realized equity curve and positions."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    buy_hold_return: float
    num_days: int
    turnover: float  # sum of |position changes| over the backtest (whipsaw proxy)
    equity_curve: pd.Series
    positions: pd.Series
    strategy_returns: pd.Series


def _max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough decline of an equity curve (<= 0)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def walk_forward(
    prices: pd.Series,
    window: int = DEFAULT_WINDOW,
    bull_threshold: float = DEFAULT_BULL_THRESHOLD,
    bear_threshold: float = DEFAULT_BEAR_THRESHOLD,
    min_history: int | None = None,
    leverage: float = 1.0,
    max_exposure: float = 1.0,
    allow_short: bool = True,
    vol_target: float | None = None,
    vol_lookback: int = 60,
    max_leverage: float = 3.0,
    hysteresis: float = 0.0,
    jump_penalty: float = 0.0,
    signal_prices: pd.Series | None = None,
    beta_align: bool = False,
    beta_lookback: int | None = None,
) -> BacktestResult:
    """Run an expanding-window walk-forward backtest over ``prices``.

    For each day we use only ``prices[: t + 1]`` to build the matrix and decide
    the position held into ``t + 1``. ``min_history`` is the number of days
    required before the first trade (defaults to ``4 * window`` so the matrix
    has a few transitions of each kind to learn from).

    If ``vol_target`` (an annualized volatility, e.g. ``0.15``) is given, the
    position is scaled by ``vol_target / trailing_realized_vol`` using only
    returns up to day ``t`` (no look-ahead) and capped at ``max_leverage`` — the
    standard volatility-targeting overlay.

    If ``signal_prices`` is given, regimes and the signal are read from *that*
    series while returns are realized on ``prices`` — i.e. trade one asset off
    another series' regime. Used for cross-asset / market-regime models, where
    the signal comes from a broad market basket rather than the asset itself.
    The two series are aligned to their common dates. With ``beta_align`` the
    signal is flipped for an asset whose trailing return correlation to the
    signal series is negative (a hedge), so a market-bull regime correctly goes
    *short* assets that rise when the market falls.
    """
    prices = prices.dropna()
    if signal_prices is None:
        signal = prices
    else:
        common = prices.index.intersection(signal_prices.dropna().index)
        prices = prices.loc[common]
        signal = signal_prices.loc[common]
    if min_history is None:
        min_history = 4 * window
    if len(prices) <= min_history + 1:
        raise ValueError(
            f"need more than {min_history + 1} price points, got {len(prices)}"
        )

    rets = prices.pct_change()
    signal_rets = signal.pct_change()
    positions: list[float] = []
    realized: list[float] = []
    index: list = []

    for t in range(min_history, len(prices) - 1):
        history = signal.iloc[: t + 1]
        states = label_states(history, window, bull_threshold, bear_threshold, jump_penalty)
        if len(states) < 2:
            continue
        P = transition_matrix(states)
        current_state = int(states.iloc[-1])
        pos = position_size(P, current_state, leverage, max_exposure, allow_short)

        if beta_align and signal_prices is not None:
            # Correlation of the asset to the signal series, no peek. Default is
            # an *expanding* window (all history up to t) for a stable,
            # structural risk-on/hedge tag; a short rolling window flips too
            # often. Flip the signal for hedges (negative correlation).
            lo = 1 if beta_lookback is None else max(1, t - beta_lookback + 1)
            corr = float(rets.iloc[lo : t + 1].corr(signal_rets.iloc[lo : t + 1]))
            if corr < 0:
                pos = -pos

        if vol_target is not None:
            # Trailing realized vol of the asset, returns up to day t only.
            recent = rets.iloc[max(1, t - vol_lookback + 1) : t + 1]
            realized_vol = float(recent.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if realized_vol > 1e-8:
                pos *= min(vol_target / realized_vol, max_leverage)
                pos = float(np.clip(pos, -max_leverage, max_leverage))

        # No-trade buffer: hold the prior position unless the target has moved
        # by more than `hysteresis` (a switch-penalty proxy that cuts whipsaw).
        if hysteresis > 0.0 and positions and abs(pos - positions[-1]) < hysteresis:
            pos = positions[-1]

        next_ret = float(rets.iloc[t + 1])
        positions.append(pos)
        realized.append(pos * next_ret)
        index.append(prices.index[t + 1])

    strategy_returns = pd.Series(realized, index=index, name="strategy_return")
    position_series = pd.Series(positions, index=index, name="position")
    # Turnover: total absolute position change (incl. initial entry from flat).
    turnover = float(position_series.diff().abs().fillna(position_series.abs()).sum())
    equity = (1.0 + strategy_returns).cumprod()
    equity.name = "equity"

    n = len(strategy_returns)
    total_return = float(equity.iloc[-1] - 1.0) if n else 0.0
    ann_factor = TRADING_DAYS_PER_YEAR / n if n else 0.0
    # A short position into a single-day move > 100% can drive equity to or
    # below zero (a blow-up). A negative base to a fractional power is complex,
    # so treat any >=100% loss as a total wipeout: annualized return = -100%.
    growth = 1.0 + total_return
    if not n:
        annualized_return = 0.0
    elif growth <= 0.0:
        annualized_return = -1.0
    else:
        annualized_return = float(growth**ann_factor - 1.0)
    ann_vol = float(strategy_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    mean_daily = float(strategy_returns.mean())
    sharpe = (
        float(mean_daily / strategy_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if strategy_returns.std(ddof=0) > 0
        else 0.0
    )

    # Buy & hold over the same traded window, for an honest comparison.
    bh_slice = prices.loc[index[0] : index[-1]] if n else prices.iloc[:0]
    buy_hold = float(bh_slice.iloc[-1] / bh_slice.iloc[0] - 1.0) if len(bh_slice) > 1 else 0.0

    return BacktestResult(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=ann_vol,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(equity) if n else 0.0,
        buy_hold_return=buy_hold,
        num_days=n,
        turnover=turnover,
        equity_curve=equity,
        positions=position_series,
        strategy_returns=strategy_returns,
    )
