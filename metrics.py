"""Statistical significance guardrails for Sharpe ratios (Bailey & López de Prado).

The Probabilistic Sharpe Ratio (PSR) gives the probability that an observed
Sharpe exceeds a benchmark once you account for sample length, skewness and
(fat-tailed) kurtosis. The Deflated Sharpe Ratio (DSR) is PSR evaluated against
the *expected maximum* Sharpe you'd see from ``n_trials`` independent backtests
of pure noise — the correction for having tried many configurations.

These operate on a return series in its native (e.g. daily) frequency; the
Sharpe inside is therefore non-annualized, which is what the formulas require.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd

_N = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


def _sharpe_moments(returns: pd.Series) -> tuple[float, float, float, int]:
    """Return (non-annualized Sharpe, skew, Pearson kurtosis, n)."""
    r = pd.Series(returns).dropna()
    n = len(r)
    sd = float(r.std(ddof=1))
    sr = float(r.mean()) / sd if sd > 0 else 0.0
    skew = float(r.skew()) if n > 2 else 0.0
    # pandas .kurtosis() is excess (Fisher); the PSR formula wants Pearson.
    kurt = float(r.kurtosis()) + 3.0 if n > 3 else 3.0
    return sr, skew, kurt, n


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > ``sr_benchmark``) given sample size, skew and kurtosis.

    ``sr_benchmark`` is in the same (non-annualized) units as the return series.
    """
    sr, skew, kurt, n = _sharpe_moments(returns)
    if n < 2:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return _N.cdf(z)


def expected_max_sharpe(sr_trials_std: float, n_trials: int) -> float:
    """Expected maximum (non-annualized) Sharpe across ``n_trials`` noise trials.

    The classic ~sqrt(ln N) selection-bias term: even pure noise produces a
    high best-of-N Sharpe, and this is how high.
    """
    if n_trials < 2 or sr_trials_std <= 0:
        return 0.0
    z1 = _N.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sr_trials_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, sr_trials_std: float
) -> float:
    """PSR against the expected-max Sharpe of ``n_trials`` trials.

    ``sr_trials_std`` is the dispersion of (non-annualized) Sharpe across the
    configurations/assets tried — the input that turns PSR into a
    multiple-testing-aware guardrail.
    """
    sr0 = expected_max_sharpe(sr_trials_std, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)
