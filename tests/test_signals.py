"""Tests for signal generation and position sizing (element 8)."""

from __future__ import annotations

import numpy as np
import pytest

from quant.signals import position_size, raw_signal
from quant.states import State


def _matrix(rows):
    return np.array(rows, dtype=float)


def test_signal_is_bull_minus_bear():
    # Row for BULL today: [bear=0.20, sideways=0.15, bull=0.65]
    P = _matrix(
        [
            [0.6, 0.3, 0.1],   # BEAR today
            [0.3, 0.4, 0.3],   # SIDEWAYS today
            [0.2, 0.15, 0.65],  # BULL today
        ]
    )
    assert raw_signal(P, State.BULL) == pytest.approx(0.65 - 0.20)
    assert raw_signal(P, State.BEAR) == pytest.approx(0.1 - 0.6)


def test_positive_signal_goes_long_negative_short():
    P = _matrix([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.2, 0.15, 0.65]])
    assert position_size(P, State.BULL) > 0    # long
    assert position_size(P, State.BEAR) < 0    # short


def test_position_is_clipped_to_max_exposure():
    P = _matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # raw signal for BULL = 1.0 - 0.0 = 1.0; leverage 5 -> clipped to max_exposure.
    assert position_size(P, State.BULL, leverage=5.0, max_exposure=1.0) == 1.0
    assert position_size(P, State.BEAR, leverage=5.0, max_exposure=0.5) == -0.5


def test_invalid_max_exposure_raises():
    P = _matrix([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.2, 0.15, 0.65]])
    with pytest.raises(ValueError):
        position_size(P, State.BULL, max_exposure=0.0)
