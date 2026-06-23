"""Element 8: turn the transition matrix into a tradeable signal.

The quant calculation is deliberately simple. Look at the one-step
probabilities *given today's state* (the relevant row of ``P``) and take:

    signal = P(bull tomorrow) - P(bear tomorrow)

* sign  -> direction: positive => go long, negative => go short
* magnitude -> conviction => position size

The raw signal lives in ``[-1, 1]``. We scale it by ``leverage`` and clip back
into ``[-max_exposure, max_exposure]`` so a strong reading sizes up while a weak
one sizes down, exactly as described ("the larger the number, the more you put
in").
"""

from __future__ import annotations

import numpy as np

from .states import State


def raw_signal(P: np.ndarray, current_state: int) -> float:
    """``P(bull) - P(bear)`` for the row corresponding to ``current_state``."""
    row = P[int(current_state)]
    return float(row[State.BULL] - row[State.BEAR])


def position_size(
    P: np.ndarray,
    current_state: int,
    leverage: float = 1.0,
    max_exposure: float = 1.0,
    allow_short: bool = True,
) -> float:
    """Target exposure for today's state.

    ``leverage`` amplifies conviction before clipping; the default of 1.0 means
    a signal of +0.45 maps to 45% long. With ``allow_short=True`` (the default)
    a negative signal becomes a net short in ``[-max_exposure, max_exposure]``;
    with ``allow_short=False`` the position floors at 0 (de-risk-only / long
    flat), which is the baseline for measuring how much shorting contributes.
    """
    if max_exposure <= 0:
        raise ValueError("max_exposure must be > 0")
    lower = -max_exposure if allow_short else 0.0
    signal = raw_signal(P, current_state) * leverage
    return float(np.clip(signal, lower, max_exposure))
