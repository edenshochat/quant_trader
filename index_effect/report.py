"""Reproducible index-effect experiment.

Loads S&P 500 additions from Wikipedia, pulls daily prices for the modern
(~2021-2026) sample from Nasdaq, benchmarks against SPY, and prints the CAR
battery, an entry-timing decomposition, a per-year breakdown, a liquidity split,
and an outlier-robustness check.

Run:  ``PYTHONPATH=<repo parent> python -m quant.index_effect.report``
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .events import load_events
from .prices import load_nasdaq
from .significance import assess
from .study import (
    align,
    car,
    liquidity_terciles,
    locate,
    study_event,
    summarize,
)

MODERN_START = dt.date(2021, 7, 1)


def _row(label: str, s: dict | None) -> str:
    if not s:
        return f"  {label:34} n=0"
    return (
        f"  {label:34} n={s['n']:3}  mean={s['mean']*100:+6.2f}%  "
        f"med={s['median']*100:+6.2f}%  t={s['t']:+5.2f}  win={s['win']*100:3.0f}%"
    )


def build(cache_wiki: str | None = None) -> tuple[list, pd.DataFrame]:
    """Fetch data and return (event-results, benchmark frame)."""
    events = [e for e in load_events(cache_wiki) if e.effective >= MODERN_START]
    spy = load_nasdaq("SPY", assetclass="etf")
    if spy is None:
        raise RuntimeError("could not load SPY benchmark")
    results, open_entries = [], {}
    for ev in events:
        px = load_nasdaq(ev.ticker)
        if px is None or len(px) < 80:
            continue
        df = align(px["close"], spy["close"])
        if len(df) < 80:
            continue
        res = study_event(df, ev.effective, ev.announcement, ev.ticker, volume=px["volume"])
        if res is None:
            continue
        # Public-info, no-foreknowledge entry: buy at the OPEN after the
        # after-close announcement, sell into the rebalance close[ED-1].
        if ev.announcement is not None:
            idx = df.index
            iA, iE = locate(idx, ev.announcement), locate(idx, ev.effective)
            if 0 < iA < iE and iA + 1 < len(df):
                o1 = px["open"].reindex(idx).iat[iA + 1]
                if o1 and not pd.isna(o1):
                    s = df["stock"].iat[iE - 1] / o1 - 1.0
                    m = df["bench"].iat[iE - 1] / df["bench"].iat[iA + 1] - 1.0
                    res.cars["buy_open1_ED1"] = float(s - m)
        results.append(res)
    return results, spy


def print_report(results: list) -> None:
    col = lambda key, rs=results: [r.cars.get(key) for r in rs]  # noqa: E731
    ad = [r for r in results if r.announcement is not None]

    print(f"\nModern S&P 500 additions analysed: {len(results)}  "
          f"(with announcement date: {len(ad)})")

    print("\n===== ANNOUNCEMENT -> REBALANCE (the classic index-effect window) =====")
    print(_row("Pre-ann drift C[AD-10]->C[AD-1]", summarize(col("ann_pre_drift", ad))))
    print(_row("Announcement day C[AD-1]->C[AD]", summarize(col("ann_day", ad))))
    print(_row("Buy C[AD] -> C[ED-1] (rebalance)", summarize(col("buy_ADclose_ED1", ad))))
    print(_row("Buy C[AD] -> C[ED]", summarize(col("buy_ADclose_ED", ad))))
    print(_row("Buy next OPEN[AD+1] -> C[ED-1]", summarize(col("buy_open1_ED1", ad))))

    print("\n===== ED-ANCHORED (no announcement date needed) =====")
    print(_row("Run-up buy C[ED-6] -> C[ED-1]", summarize(col("runup_ED6_ED1"))))
    print(_row("Run-up buy C[ED-10] -> C[ED-1]", summarize(col("runup_ED10_ED1"))))
    print(_row("Reversal C[ED-1] -> C[ED+10]", summarize(col("reversal_ED1_ED10"))))
    print(_row("Reversal C[ED-1] -> C[ED+20]", summarize(col("reversal_ED1_ED20"))))

    print("\n===== PER CALENDAR YEAR (buy C[AD] -> C[ED-1]) =====")
    for y in range(2021, 2027):
        sub = [r for r in ad if r.effective.year == y]
        if sub:
            print(_row(f"  {y}", summarize([r.cars.get("buy_ADclose_ED1") for r in sub])))

    print("\n===== OUTLIER ROBUSTNESS (buy C[AD] -> C[ED-1]) =====")
    xs = sorted(v for v in col("buy_ADclose_ED1", ad) if v is not None)
    print(_row("all", summarize(xs)))
    print(_row("drop top 3", summarize(xs[:-3])))
    print(_row("trim 3 each tail", summarize(xs[3:-3])))

    print("\n===== LIQUIDITY SPLIT (short the post-rebalance reversal) =====")
    terciles = liquidity_terciles(results)
    for name, bucket in terciles:
        if not bucket:
            continue
        lo, hi = bucket[0].dollar_volume / 1e6, bucket[-1].dollar_volume / 1e6
        s = summarize([r.cars.get("reversal_ED1_ED20") for r in bucket])
        short = -s["mean"] * 100 if s else float("nan")
        print(f"  {name:9} (${lo:6.0f}M..${hi:7.0f}M/day)  "
              f"reversal long={s['mean']*100:+6.2f}% => SHORT P&L={short:+6.2f}%  t={s['t']:+5.2f}  n={s['n']}")

    print("\n===== MULTIPLE-TESTING-AWARE SIGNIFICANCE (repo PSR / DSR) =====")
    low_tercile = terciles[0][1]
    strategies = {
        "LONG public-info O[AD+1]->ED-1": [r.cars.get("buy_open1_ED1") for r in ad],
        "LONG prompt C[AD]->ED-1": [r.cars.get("buy_ADclose_ED1") for r in ad],
        "SHORT illiquid reversal ED-1->+20": [
            -r.cars["reversal_ED1_ED20"] for r in low_tercile
        ],
    }
    assessed = assess(strategies)
    for name, a in assessed.items():
        print(f"  {name:34} n={a['n']:3}  Sharpe/trade={a['sharpe']:+.2f}  "
              f"PSR={a['psr']:.3f}  DSR={a['dsr']:.3f}")
    print("  (DSR>0.95 => the edge survives correction for the windows examined)")


def main() -> None:
    results, _ = build()
    print_report(results)


if __name__ == "__main__":
    main()
