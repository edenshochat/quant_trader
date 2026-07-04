"""Cross-index validation: run the same event-study logic on every index in
``events.INDEX_PAGES`` to see whether the inclusion edge generalises beyond the
S&P 500.

Each index's additions are benchmarked against **its own tracking ETF** (so the
CAR is stock-minus-index), over the Nasdaq-covered modern window (~2021→now).
The headline metric is the announcement-free **run-up** ``C[ED-6]→C[ED-1]``
(every change table has effective dates), plus the post-rebalance reversal and,
where citation announcement dates exist, the ``C[AD]→C[ED-1]`` capture.

Hypothesis: the edge should be *larger* in smaller-cap indices (S&P 600, mid),
where index demand is bigger relative to float — the low-liquidity effect.
"""

from __future__ import annotations

import datetime as dt
import os

from .events import INDEX_PAGES, load_events
from .prices import load_nasdaq
from .study import align, study_event, summarize

MODERN_START = dt.date(2021, 7, 1)
_TICKER_OK = __import__("re").compile(r"^[A-Z][A-Z.\-]{0,5}$")


def run_index(page: str, etf: str, wiki_cache: str | None = None,
              start: dt.date = MODERN_START, max_events: int = 100) -> list:
    """Event results for one index's additions vs its own ETF benchmark.

    ``max_events`` caps to the most recent N additions (churny small-cap indices
    have hundreds; a recent sample is ample for the statistics and keeps the
    scan fast).
    """
    bench = load_nasdaq(etf, assetclass="etf")
    if bench is None:
        return []
    events = [
        e for e in load_events(page, wiki_cache)
        if e.effective >= start and _TICKER_OK.match(e.ticker)
    ]
    events = sorted(events, key=lambda e: e.effective)[-max_events:]
    seen, results = set(), []
    for ev in events:
        if (ev.ticker, ev.effective) in seen:
            continue
        seen.add((ev.ticker, ev.effective))
        px = load_nasdaq(ev.ticker, tries=2, fallback=False)
        if px is None or len(px) < 80:
            continue
        df = align(px["close"], bench["close"])
        if len(df) < 80:
            continue
        res = study_event(df, ev.effective, ev.announcement, ev.ticker, volume=px["volume"])
        if res is not None:
            results.append(res)
    return results


def _fmt(name: str, etf: str, res: list) -> str:
    def col(k, rs=res):
        return [r.cars.get(k) for r in rs]
    ad = [r for r in res if r.announcement is not None]
    ru = summarize(col("runup_ED6_ED1"))
    rv = summarize(col("reversal_ED1_ED20"))
    adc = summarize(col("buy_ADclose_ED1", ad))

    def cell(s):
        return f"{s['mean']*100:+5.2f}% t={s['t']:+4.1f}" if s else "   n/a    "
    return (f"  {name:16} {etf:4} n={len(res):3}  "
            f"runup[ED-6→ED-1] {cell(ru)}   "
            f"reversal[→ED+20] {cell(rv)}   "
            f"buyAD→ED-1 {cell(adc)} (n_ad={len(ad)})")


def main(cache_dir: str | None = None) -> None:
    print("Cross-index validation — additions vs own-ETF benchmark, ~2021→now")
    print("(run-up = the announcement-free tradable window; reversal SHORT = -reversal)\n")
    for name, (page, etf) in INDEX_PAGES.items():
        wc = None
        if cache_dir:
            wc = os.path.join(cache_dir, f"_wiki_{etf}.json")
        res = run_index(page, etf, wiki_cache=wc)
        print(_fmt(name, etf, res))


if __name__ == "__main__":
    main(cache_dir=os.path.dirname(__file__))
