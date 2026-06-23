"""Tests for the transition matrix and its derivatives (elements 4-7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.states import NUM_STATES, State, label_states
from quant.transition import (
    count_transitions,
    n_step,
    stationary_distribution,
    stickiness,
    transition_matrix,
)


def test_rows_sum_to_one(noisy_prices):
    states = label_states(noisy_prices, window=20)
    P = transition_matrix(states)
    assert P.shape == (NUM_STATES, NUM_STATES)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(NUM_STATES))


def test_counts_match_known_sequence():
    # BULL,BULL,BEAR,BULL  -> transitions: B->B, B->Be, Be->B
    seq = pd.Series([State.BULL, State.BULL, State.BEAR, State.BULL])
    counts = count_transitions(seq)
    assert counts[State.BULL, State.BULL] == 1
    assert counts[State.BULL, State.BEAR] == 1
    assert counts[State.BEAR, State.BULL] == 1
    assert counts.sum() == 3


def test_empty_row_falls_back_to_uniform():
    # Only BULL/BEAR observed; SIDEWAYS row has no outgoing transitions.
    seq = pd.Series([State.BULL, State.BEAR, State.BULL, State.BEAR])
    with pytest.warns(UserWarning):
        P = transition_matrix(seq)
    np.testing.assert_allclose(P[State.SIDEWAYS], np.full(NUM_STATES, 1 / NUM_STATES))
    np.testing.assert_allclose(P.sum(axis=1), np.ones(NUM_STATES))


def test_n_step_equals_matrix_power(noisy_prices):
    states = label_states(noisy_prices, window=20)
    P = transition_matrix(states)
    np.testing.assert_allclose(n_step(P, 3), P @ P @ P)


def test_n_step_validates_n(noisy_prices):
    P = transition_matrix(label_states(noisy_prices, window=20))
    with pytest.raises(ValueError):
        n_step(P, 0)


def test_stickiness_is_diagonal(noisy_prices):
    P = transition_matrix(label_states(noisy_prices, window=20))
    np.testing.assert_array_equal(stickiness(P), np.diag(P))


def test_stationary_distribution_is_fixed_point(noisy_prices):
    P = transition_matrix(label_states(noisy_prices, window=20))
    pi = stationary_distribution(P)
    assert pytest.approx(pi.sum(), abs=1e-9) == 1.0
    assert (pi >= 0).all()
    # pi @ P == pi  (the defining property)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-8)


def test_high_horizon_converges_to_stationary(noisy_prices):
    # A well-mixing (irreducible, aperiodic) chain: every row of a high power
    # collapses to the stationary distribution. Slow-mixing/near-absorbing
    # chains converge too, just over many more steps — hence the noisy fixture.
    P = transition_matrix(label_states(noisy_prices, window=20))
    pi = stationary_distribution(P)
    far = n_step(P, 5000)
    for i in range(NUM_STATES):
        np.testing.assert_allclose(far[i], pi, atol=1e-4)
