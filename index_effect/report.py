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

import statistics as st

from .events import load_events
from .portfolio import (
    build_trade,
    longest_underwater,
    max_drawdown,
    simulate,
    worst_losing_streak,
)
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


def print_pocket(spy) -> None:
    """$100k pocket P&L and drawdown over the covered (Nasdaq ~5y) window."""
    events = [
        e for e in load_events() if e.effective >= dt.date(2020, 1, 1) and e.announcement
    ]
    trades = []
    for ev in events:
        px = load_nasdaq(ev.ticker)
        if px is None or len(px) < 10:
            continue
        t = build_trade(px, ev.announcement, ev.effective, ev.ticker)
        if t is not None:
            trades.append(t)
    if not trades:
        print("\n(no pocket sim — no covered trades)")
        return
    trades.sort(key=lambda t: t.entry_date)
    first, last = trades[0].entry_date, max(t.exit_date for t in trades)
    calendar = [d.date().isoformat() for d in spy.index
                if first <= d.date().isoformat() <= last]

    curve, taken, skipped = simulate(trades, calendar, frac=1.0)
    rets = [t.ret for t in taken]
    dd = max_drawdown(curve)
    yrs = (dt.date.fromisoformat(curve[-1][0]) - dt.date.fromisoformat(curve[0][0])).days / 365.25
    final = curve[-1][1]
    invested = sum(1 for d, _ in curve if any(t.entry_date <= d <= t.exit_date for t in taken))
    streak, streak_names = worst_losing_streak(taken)

    print(f"\n===== $100k POCKET — concrete P&L & drawdown ({curve[0][0]} → {curve[-1][0]}) =====")
    print(f"  entry OPEN[AD+1], exit CLOSE[ED-1], no leverage, no slippage")
    print(f"  trades: {len(taken)} taken, {len(skipped)} skipped (overlap)   "
          f"invested ~{invested/len(curve)*100:.0f}% of the time")
    print(f"  final ${final:,.0f}  ({final/100000-1:+.0%}, CAGR {(final/100000)**(1/yrs)-1:+.1%})   "
          f"win {sum(r>0 for r in rets)/len(rets)*100:.0f}%  avg {st.mean(rets)*100:+.2f}%")
    print(f"  worst trade {min(rets)*100:+.1f}%   worst losing streak {streak*100:+.1f}% "
          f"({'+'.join(streak_names)})")
    print(f"  >> MAX DRAWDOWN {dd['mdd']*100:.1f}%  (${dd['peak']:,.0f} {dd['peak_date']} "
          f"→ ${dd['trough']:,.0f} {dd['trough_date']})   underwater up to "
          f"{longest_underwater(curve)} trading days")
    print("  position-size dial (fraction of pocket per trade):")
    for f in (1.0, 0.5, 0.25):
        c, _, _ = simulate(trades, calendar, frac=f)
        print(f"    {f*100:3.0f}%  final ${c[-1][1]:>9,.0f}  ({c[-1][1]/100000-1:+6.0%})  "
              f"maxDD {max_drawdown(c)['mdd']*100:6.1f}%")


def main() -> None:
    results, spy = build()
    print_report(results)
    print_pocket(spy)


if __name__ == "__main__":
    main()
