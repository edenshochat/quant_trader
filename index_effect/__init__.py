"""S&P 500 "index effect" event study.

An empirical test of the common wisdom that index-inclusion front-running has
been fully arbitraged away. See ``index_effect/FINDINGS.md`` for results.

NOT financial advice — an educational research reproduction.
"""

from __future__ import annotations

from .events import AdditionEvent, load_events, parse_changes
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
    "load_events",
    "parse_changes",
    "EventResult",
    "align",
    "car",
    "locate",
    "study_event",
    "summarize",
    "liquidity_terciles",
]
