"""Orchestrator that ties the 10 elements into one analysis.

``MarkovRegimeModel`` holds the parameters (ticker, look-back window, regime
thresholds) and ``analyze()`` produces a :class:`RegimeReport` with everything
the transcript walks through: the transition matrix, stickiness, multi-horizon
forecasts, the stationary distribution, today's state and signal, the
walk-forward backtest, and the optional HMM cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, walk_forward
from .data import load_prices
from .signals import position_size, raw_signal
from .states import (
    DEFAULT_BEAR_THRESHOLD,
    DEFAULT_BULL_THRESHOLD,
    DEFAULT_WINDOW,
    State,
    label_states,
)
from .transition import (
    n_step,
    stationary_distribution,
    stickiness,
    transition_matrix,
)

DEFAULT_HORIZONS = (1, 2, 3, 28)


@dataclass
class RegimeReport:
    ticker: str
    window: int
    bull_threshold: float
    bear_threshold: float
    num_labeled_days: int

    transition_matrix: np.ndarray
    stickiness: np.ndarray
    stationary_distribution: np.ndarray
    horizon_forecasts: dict[int, np.ndarray]

    current_state: int
    current_signal: float
    current_position: float

    backtest: BacktestResult | None = None
    hmm: object | None = None  # HMMResult, kept loose to avoid hard hmmlearn import
    extras: dict = field(default_factory=dict)

    @property
    def current_state_name(self) -> str:
        return State(self.current_state).name.capitalize()


class MarkovRegimeModel:
    """The observable Markov regime model from the "hedge fund method"."""

    def __init__(
        self,
        ticker: str | None = None,
        window: int = DEFAULT_WINDOW,
        bull_threshold: float = DEFAULT_BULL_THRESHOLD,
        bear_threshold: float = DEFAULT_BEAR_THRESHOLD,
    ) -> None:
        self.ticker = ticker
        self.window = window
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def analyze(
        self,
        prices: pd.Series | None = None,
        period: str = "10y",
        csv: str | None = None,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        run_backtest: bool = True,
        run_hmm: bool = True,
        leverage: float = 1.0,
        max_exposure: float = 1.0,
        allow_short: bool = True,
        vol_target: float | None = None,
        max_leverage: float = 3.0,
        hysteresis: float = 0.0,
        jump_penalty: float = 0.0,
    ) -> RegimeReport:
        if prices is None:
            prices = load_prices(self.ticker, period=period, csv=csv)
        label = self.ticker or getattr(prices, "name", None) or "series"

        states = label_states(
            prices, self.window, self.bull_threshold, self.bear_threshold, jump_penalty
        )
        if len(states) < 2:
            raise ValueError(
                "not enough labeled days to build a transition matrix "
                f"(got {len(states)}); supply more price history"
            )

        P = transition_matrix(states)
        current_state = int(states.iloc[-1])

        report = RegimeReport(
            ticker=str(label),
            window=self.window,
            bull_threshold=self.bull_threshold,
            bear_threshold=self.bear_threshold,
            num_labeled_days=len(states),
            transition_matrix=P,
            stickiness=stickiness(P),
            stationary_distribution=stationary_distribution(P),
            horizon_forecasts={n: n_step(P, n) for n in horizons},
            current_state=current_state,
            current_signal=raw_signal(P, current_state),
            current_position=position_size(
                P, current_state, leverage, max_exposure, allow_short
            ),
        )

        if run_backtest:
            try:
                report.backtest = walk_forward(
                    prices,
                    self.window,
                    self.bull_threshold,
                    self.bear_threshold,
                    leverage=leverage,
                    max_exposure=max_exposure,
                    allow_short=allow_short,
                    vol_target=vol_target,
                    max_leverage=max_leverage,
                    hysteresis=hysteresis,
                    jump_penalty=jump_penalty,
                )
            except ValueError as exc:
                report.extras["backtest_error"] = str(exc)

        if run_hmm:
            try:
                from .hmm import cross_check  # noqa: PLC0415

                report.hmm = cross_check(
                    prices,
                    self.window,
                    self.bull_threshold,
                    self.bear_threshold,
                )
            except ImportError as exc:
                report.extras["hmm_error"] = str(exc)

        return report
