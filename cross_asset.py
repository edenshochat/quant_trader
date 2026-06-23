"""Cross-asset / macro regime models.

The single-asset blindspot: defining a regime from one asset's *own* noisy
returns. Practitioners instead read the regime from a broad basket (or macro
variables), because risk-on/risk-off shifts move the whole cross-section at
once and a basket averages out idiosyncratic noise into a cleaner, more
persistent signal.

``build_market_index`` collapses a basket into one equal-weight composite
"market" price series; ``walk_forward(..., signal_prices=market)`` then trades
any target asset off that shared regime instead of its own.
"""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult, walk_forward

# A broad risk-on basket spanning equities, credit, and commodities — a proxy
# for "the market environment" rather than any single asset.
DEFAULT_MARKET_BASKET = ("SPY", "EEM", "EFA", "HYG", "GLD", "DBC")


def build_market_index(basket: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight composite of a basket, rebased to 1.0 on the common start.

    Aligns the members to their overlapping dates (inner join), rebases each to
    1.0, and averages — a synthetic broad-market price series.
    """
    if not basket:
        raise ValueError("basket is empty")
    df = pd.DataFrame(basket).dropna()
    if len(df) < 2:
        raise ValueError("basket members have too little overlapping history")
    rebased = df / df.iloc[0]
    index = rebased.mean(axis=1)
    index.name = "MARKET"
    return index


def own_vs_market(
    target_prices: pd.Series,
    market_index: pd.Series,
    beta_align: bool = False,
    **kwargs,
) -> tuple[BacktestResult, BacktestResult]:
    """Backtest a target asset off its OWN regime vs the shared MARKET regime.

    ``kwargs`` (window, thresholds, vol_target, jump_penalty, ...) are passed to
    both backtests so only the regime *source* differs. ``beta_align`` flips the
    market signal for hedges (negative trailing correlation to the market).
    """
    own = walk_forward(target_prices, **kwargs)
    market = walk_forward(
        target_prices, signal_prices=market_index, beta_align=beta_align, **kwargs
    )
    return own, market
