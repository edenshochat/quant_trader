"""Multiple-testing-aware significance for the index-effect strategies.

A single-window t-stat overstates confidence when several windows were examined.
We reuse the repo's own guardrails (:mod:`quant.metrics`): the Probabilistic
Sharpe Ratio (PSR, P[true Sharpe > 0]) and the Deflated Sharpe Ratio (DSR, PSR
against the expected best-of-N Sharpe from pure-noise trials). Each addition's
trade is one observation in the return series; the per-trade Sharpe is therefore
non-annualized, which is what these formulas require.
"""

from __future__ import annotations

import statistics as st

import pandas as pd

from ..metrics import deflated_sharpe_ratio, probabilistic_sharpe_ratio


def per_trade_sharpe(returns) -> float:
    xs = [float(x) for x in returns if x is not None]
    if len(xs) < 2:
        return 0.0
    sd = st.pstdev(xs)
    return (st.mean(xs) / sd) if sd > 0 else 0.0


def assess(
    strategies: dict[str, list[float]], n_trials: int = 10
) -> dict[str, dict]:
    """PSR/DSR for each named strategy return series.

    ``strategies`` maps a label to its per-trade return list. ``n_trials`` is the
    (approximate) number of window/entry configurations examined across the
    study; the dispersion of per-trade Sharpe across the supplied strategies is
    used as the selection-bias ``sr_trials_std`` input to the DSR.
    """
    clean = {k: [float(x) for x in v if x is not None] for k, v in strategies.items()}
    sharpes = [per_trade_sharpe(v) for v in clean.values() if len(v) >= 2]
    sr_std = st.pstdev(sharpes) if len(sharpes) > 1 else 0.1
    out: dict[str, dict] = {}
    for name, xs in clean.items():
        if len(xs) < 2:
            out[name] = {"n": len(xs), "sharpe": 0.0, "psr": float("nan"), "dsr": float("nan")}
            continue
        r = pd.Series(xs)
        out[name] = {
            "n": len(xs),
            "sharpe": per_trade_sharpe(xs),
            "psr": probabilistic_sharpe_ratio(r, 0.0),
            "dsr": deflated_sharpe_ratio(r, n_trials=n_trials, sr_trials_std=sr_std),
            "sr_trials_std": sr_std,
        }
    return out
