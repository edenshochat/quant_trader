"""Markov regime "hedge fund method" quant engine.

An educational reproduction of the 10-element Markov regime trading method:
quantify the market into bull/sideways/bear states, build a transition matrix,
forecast forward, generate a signal, and validate with a walk-forward backtest
and an HMM cross-check. See README.md.

NOT financial advice.
"""

from __future__ import annotations

from .regime import MarkovRegimeModel, RegimeReport
from .states import State, label_states
from .transition import (
    n_step,
    stationary_distribution,
    stickiness,
    transition_matrix,
)

__all__ = [
    "MarkovRegimeModel",
    "RegimeReport",
    "State",
    "label_states",
    "transition_matrix",
    "n_step",
    "stationary_distribution",
    "stickiness",
]
