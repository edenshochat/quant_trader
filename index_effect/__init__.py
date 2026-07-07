"""S&P 500 "index effect" event study.

An empirical test of the common wisdom that index-inclusion front-running has
been fully arbitraged away. See ``index_effect/FINDINGS.md`` for results.

NOT financial advice — an educational research reproduction.
"""

from __future__ import annotations

from .cross_index import run_index
from .events import INDEX_PAGES, AdditionEvent, load_events, parse_changes
from .portfolio import (
    Trade,
    build_trade,
    combine_curves,
    longest_underwater,
    max_drawdown,
    simulate,
    worst_losing_streak,
)
from .significance import assess, per_trade_sharpe
from .study import (
    EventResult,
    align,
    car,
    liquidity_terciles,
    locate,
    study_event,
    summarize,
)

__all__ = [
    "AdditionEvent",
    "INDEX_PAGES",
    "run_index",
    "load_events",
    "parse_changes",
    "EventResult",
    "align",
    "car",
    "locate",
    "study_event",
    "summarize",
    "liquidity_terciles",
    "assess",
    "per_trade_sharpe",
    "Trade",
    "build_trade",
    "simulate",
    "combine_curves",
    "max_drawdown",
    "longest_underwater",
    "worst_losing_streak",
]
