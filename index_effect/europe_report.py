"""European (DAX/TecDAX) index-inclusion event study — same methodology as the
US indices, run on the official STOXX historical-compositions data (see
europe.py for why Wikipedia doesn't work for Europe).

Run: ``PYTHONPATH=<repo parent> python -m quant.index_effect.europe_report``
"""

from __future__ import annotations

import datetime as dt
import os

from .europe import INDEX_CONFIG, load_events
from .portfolio import build_trade, max_drawdown, simulate
from .prices import load_yahoo
from .significance import assess
from .study import align, car, locate, study_event, summarize

PDF_CACHE = os.path.join(os.path.dirname(__file__), "_stoxx_historical_compositions.pdf")

# Fixed, wide benchmark window shared by every call so the on-disk cache (keyed
# by exact ticker+start+end) hits reliably instead of drifting per-index.
_BENCH_START, _BENCH_END = dt.date(2017, 1, 1), dt.date(2026, 7, 1)


def _stock_window(events, ticker):
    """Per-ticker window covering every one of its appearances (+/- 60d) --
    tickers that recur across multiple reconstitutions share one cached fetch."""
    all_eff = [e for e, _, t in events if t == ticker]
    return min(all_eff) - dt.timedelta(days=60), max(all_eff) + dt.timedelta(days=60)


def build(index: str):
    """Fetch data and return (event-results, benchmark frame) for one index."""
    _, bench_ticker = INDEX_CONFIG[index]
    events = load_events(index, PDF_CACHE)
    if not events:
        return [], None
    bench = load_yahoo(bench_ticker, _BENCH_START, _BENCH_END)
    if bench is None:
        return [], None
    results = []
    for eff, ann, ticker in events:
        lo, hi = _stock_window(events, ticker)
        px = load_yahoo(ticker, lo, hi)
        if px is None or len(px) < 40:
            continue
        df = align(px["close"], bench["close"])
        if len(df) < 40:
            continue
        res = study_event(df, eff, ann, ticker, volume=px["volume"], pre_lo=-25, pre_hi=-10, pad=15)
        if res is None:
            continue
        results.append(res)
    return results, bench


def _row(label, s):
    if not s:
        return f"  {label:34} n=0"
    return (f"  {label:34} n={s['n']:3}  mean={s['mean']*100:+6.2f}%  "
            f"med={s['median']*100:+6.2f}%  t={s['t']:+5.2f}  win={s['win']*100:3.0f}%")


def print_report(index: str, results: list) -> None:
    ad = [r for r in results if r.announcement is not None]
    print(f"\n===== {index}: {len(results)} additions analysed ({len(ad)} with announcement) =====")
    print(_row("Announcement day C[AD-1]->C[AD]", summarize([r.cars.get("ann_day") for r in ad])))
    print(_row("Buy C[AD] -> C[ED-1] (rebalance)", summarize([r.cars.get("buy_ADclose_ED1") for r in ad])))
    print(_row("Run-up C[ED-6] -> C[ED-1]", summarize([r.cars.get("runup_ED6_ED1") for r in results])))
    print(_row("Reversal C[ED-1] -> C[ED+20]", summarize([r.cars.get("reversal_ED1_ED20") for r in results])))
    print(_row("Pre-ann drift C[AD-10]->C[AD-1]", summarize([r.cars.get("ann_pre_drift") for r in ad])))


def print_pocket(index: str, results: list, bench) -> None:
    events = load_events(index, PDF_CACHE)
    trades = []
    for eff, ann, ticker in events:
        lo, hi = _stock_window(events, ticker)
        px = load_yahoo(ticker, lo, hi)
        if px is None:
            continue
        t = build_trade(px, ann, eff, ticker)
        if t is not None:
            trades.append(t)
    if not trades:
        print(f"  {index}: no pocket trades")
        return
    trades.sort(key=lambda t: t.entry_date)
    first, last = trades[0].entry_date, max(t.exit_date for t in trades)
    calendar = sorted(d for d in {d.date().isoformat() for d in bench.index} if first <= d <= last)
    curve, taken, skipped = simulate(trades, calendar, frac=1.0, start_cash=100_000.0)
    dd = max_drawdown(curve)
    rets = [t.ret for t in taken]
    win = sum(r > 0 for r in rets) / len(rets) if rets else float("nan")
    print(f"  {index:8} n={len(taken):3}  final ${curve[-1][1]:>10,.0f} "
          f"({curve[-1][1]/1e5-1:+7.1%})  maxDD {dd['mdd']*100:6.1f}%  win={win*100:.0f}%  "
          f"window {first}->{last}")


def main() -> None:
    all_results = {}
    for index in INDEX_CONFIG:
        results, bench = build(index)
        all_results[index] = (results, bench)
        print_report(index, results)

    print("\n===== $100k POCKET (buy OPEN[AD+1], sell CLOSE[ED-1], all-in) =====")
    for index, (results, bench) in all_results.items():
        if bench is not None:
            print_pocket(index, results, bench)

    print("\n===== SIGNIFICANCE (repo PSR/DSR) =====")
    strategies = {}
    for index, (results, _) in all_results.items():
        ad = [r for r in results if r.announcement is not None]
        strategies[index] = [r.cars.get("buy_ADclose_ED1") for r in ad]
    assessed = assess(strategies, n_trials=6)
    for name, a in assessed.items():
        print(f"  {name:10} n={a['n']:3}  Sharpe/trade={a['sharpe']:+.2f}  "
              f"PSR={a['psr']:.3f}  DSR={a['dsr']:.3f}")


if __name__ == "__main__":
    main()
