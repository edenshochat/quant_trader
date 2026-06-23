"""Tests for the HMM cross-check (element 10).

Skipped automatically when hmmlearn is not installed so the core suite still
runs offline without the optional dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hmmlearn")

from quant.hmm import cross_check, fit_hmm_states  # noqa: E402
from quant.states import State  # noqa: E402


def test_hmm_recovers_and_orders_regimes(regime_switch_prices):
    hmm_states, means = fit_hmm_states(regime_switch_prices)
    # Means must be sorted ascending (BEAR < SIDEWAYS < BULL) after remapping.
    assert np.all(np.diff(means) >= 0)
    # Both a bull-ish and a bear-ish regime should be present.
    present = set(hmm_states.unique())
    assert State.BULL in present
    assert State.BEAR in present


def test_first_half_bull_second_half_bear(regime_switch_prices):
    hmm_states, _ = fit_hmm_states(regime_switch_prices)
    half = len(hmm_states) // 2
    first, second = hmm_states.iloc[:half], hmm_states.iloc[half:]
    # The early bull block should be more bullish than the late bear block.
    assert first.mean() > second.mean()


def test_cross_check_reports_agreement(regime_switch_prices):
    result = cross_check(regime_switch_prices, window=20)
    assert 0.0 <= result.agreement_rate <= 1.0
    assert len(result.confirmation) == len(result.hmm_states)
    # Strongly separated regimes => rule labels and HMM should mostly agree.
    assert result.agreement_rate > 0.5
