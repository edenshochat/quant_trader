"""Pooling the index-inclusion trade across index families — does diversifying
capital across S&P 500/400/600 + Nasdaq-100 reduce the single-index pocket's
drawdown, and does weighting sleeves by their statistical edge (DSR) do better
than splitting capital evenly?

Each index gets its own **sleeve**: a slice of the $100k pocket that trades only
that index's announcement-dated additions, sequentially and all-in (one name at
a time within the sleeve — reusing :func:`portfolio.simulate` unchanged). Sleeve
equity curves are summed (:func:`portfolio.combine_curves`) into one portfolio
curve. This is deliberately the simplest diversification model: no shared
capital contention between sleeves, so any drawdown improvement over a single
concentrated index is coming purely from **event/return correlation being less
than 1** across index families, not from a sizing trick.

Restricted to the Nasdaq-covered window (~2021-07 onward) so it runs on cached
data without needing the rate-limited Yahoo fallback — see ``FINDINGS.md`` for
the 2020-covering single-index number this compares against.
"""

from __future__ import annotations

import datetime as dt
import os
import statistics as st

from .events import INDEX_PAGES, load_events
from .portfolio import (
    Trade,
    build_hedged_trade,
    build_trade,
    build_trade_prompt,
    combine_curves,
    max_drawdown,
    simulate,
)
from .prices import load_nasdaq
from .significance import assess

NASDAQ_START = dt.date(2021, 7, 1)


def build_index_trades(page: str, etf: str, wiki_cache: str | None = None,
                       start: dt.date = NASDAQ_START, hedged: bool = False,
                       entry: str = "open_next") -> list[Trade]:
    """Announcement-dated addition trades for one index, Nasdaq-only window.

    ``hedged=True`` builds each trade market-neutral (long stock, short the
    index's own tracking ETF) via :func:`portfolio.build_hedged_trade`.

    ``entry`` picks the entry convention:
      - ``"open_next"`` (default) — realistic, no-foreknowledge: buy the open
        the session after S&P's after-close announcement.
      - ``"close_same"`` — buy the announcement day's close itself
        (:func:`portfolio.build_trade_prompt`); captures the overnight gap but
        requires same-day execution. Not compatible with ``hedged=True``.
    """
    events = [
        e for e in load_events(page, wiki_cache)
        if e.effective >= start and e.announcement and e.announcement >= start
    ]
    bench = load_nasdaq(etf, assetclass="etf") if hedged else None
    if hedged and bench is None:
        return []
    trades = []
    for ev in events:
        px = load_nasdaq(ev.ticker, tries=2, fallback=False)
        if px is None or len(px) < 10:
            continue
        if hedged:
            t = build_hedged_trade(px, bench, ev.announcement, ev.effective, ev.ticker)
        elif entry == "close_same":
            t = build_trade_prompt(px, ev.announcement, ev.effective, ev.ticker)
        else:
            t = build_trade(px, ev.announcement, ev.effective, ev.ticker)
        if t is not None:
            trades.append(t)
    trades.sort(key=lambda t: t.entry_date)
    return trades


def equal_weights(names: list[str]) -> dict[str, float]:
    return {n: 1.0 / len(names) for n in names}


def dsr_weights(trades_by_index: dict[str, list[Trade]], floor: float = 0.05) -> dict[str, float]:
    """Weight each sleeve by its own Deflated Sharpe Ratio.

    Uses each index's *own* realized trade returns as the "strategy" fed to
    :func:`significance.assess` (n_trials = number of sleeves, since that's the
    family of configurations being compared here). Weights are DSR clipped to
    ``[floor, 1]`` and renormalized — a sleeve with a weak/insignificant edge
    still gets a token allocation (``floor``) rather than zero, since DSR on
    ~40-170 trades is itself a noisy estimate.
    """
    names = list(trades_by_index)
    strategies = {n: [t.ret for t in trades_by_index[n]] for n in names}
    scored = assess(strategies, n_trials=len(names))
    raw = {n: max(scored[n]["dsr"], floor) if scored[n]["n"] >= 2 else floor for n in names}
    total = sum(raw.values())
    return {n: v / total for n, v in raw.items()}


def simulate_pooled(trades_by_index: dict[str, list[Trade]], calendar: list[str],
                    weights: dict[str, float], start_cash: float = 100_000.0) -> dict:
    """Run one sleeve per index at its weighted share of capital, sum the curves."""
    sleeve_curves, sleeve_taken = {}, {}
    for name, trades in trades_by_index.items():
        c, taken, _ = simulate(trades, calendar, frac=1.0, start_cash=start_cash * weights[name])
        sleeve_curves[name] = c
        sleeve_taken[name] = taken
    combined = combine_curves(list(sleeve_curves.values()))
    return {"combined": combined, "sleeves": sleeve_curves, "taken": sleeve_taken}


def sleeve_correlation(sleeve_curves: dict[str, list[tuple[str, float]]]) -> dict:
    """Pairwise correlation of sleeves' daily returns — the diversification check."""
    def daily_rets(curve):
        vals = [e for _, e in curve]
        return [vals[i] / vals[i - 1] - 1.0 for i in range(1, len(vals))]

    names = list(sleeve_curves)
    rets = {n: daily_rets(sleeve_curves[n]) for n in names}
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xs, ys = rets[a], rets[b]
            if len(xs) >= 2 and st.pstdev(xs) > 0 and st.pstdev(ys) > 0:
                out[f"{a} / {b}"] = _corr(xs, ys)
    return out


def _corr(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def _wiki_cache_path(etf: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"_wiki_{etf}.json")


def main() -> None:
    per_index = {
        name: build_index_trades(page, etf, wiki_cache=_wiki_cache_path(etf))
        for name, (page, etf) in INDEX_PAGES.items()
    }
    for name, trades in per_index.items():
        print(f"  {name:16} {len(trades)} announcement-dated trades since {NASDAQ_START}")

    all_trades = [t for ts in per_index.values() for t in ts]
    first, last = min(t.entry_date for t in all_trades), max(t.exit_date for t in all_trades)
    spy = load_nasdaq("SPY", assetclass="etf")
    calendar = sorted(d for d in {d.date().isoformat() for d in spy.index} if first <= d <= last)

    print(f"\n===== BASELINE: single-index, all-in ({first} → {last}) =====")
    for name, trades in per_index.items():
        c, taken, _ = simulate(trades, calendar, frac=1.0, start_cash=100_000.0)
        dd = max_drawdown(c)
        print(f"  {name:16} final ${c[-1][1]:>10,.0f} ({c[-1][1]/1e5-1:+7.1%})  "
              f"maxDD {dd['mdd']*100:6.1f}%  n={len(taken)}")

    print("\n===== POOLED: equal-weight sleeves (25% each) =====")
    eq = simulate_pooled(per_index, calendar, equal_weights(list(per_index)))
    dd_eq = max_drawdown(eq["combined"])
    print(f"  combined final ${eq['combined'][-1][1]:>10,.0f} "
          f"({eq['combined'][-1][1]/1e5-1:+7.1%})  maxDD {dd_eq['mdd']*100:6.1f}%")
    for name, c in eq["sleeves"].items():
        print(f"    sleeve {name:16} final ${c[-1][1]:>9,.0f}  "
              f"maxDD {max_drawdown(c)['mdd']*100:6.1f}%  n={len(eq['taken'][name])}")

    print("\n===== POOLED: DSR-weighted sleeves =====")
    w = dsr_weights(per_index)
    for name, weight in w.items():
        print(f"  weight {name:16} {weight*100:5.1f}%")
    dw = simulate_pooled(per_index, calendar, w)
    dd_dw = max_drawdown(dw["combined"])
    print(f"  combined final ${dw['combined'][-1][1]:>10,.0f} "
          f"({dw['combined'][-1][1]/1e5-1:+7.1%})  maxDD {dd_dw['mdd']*100:6.1f}%")

    print("\n===== SLEEVE CORRELATION (daily returns) — the diversification check =====")
    for pair, r in sleeve_correlation(eq["sleeves"]).items():
        print(f"  {pair:32} r={r:+.2f}")

    print("\n===== HEDGED: long stock / short own-index ETF (strips market beta) =====")
    per_index_hedged = {
        name: build_index_trades(page, etf, wiki_cache=_wiki_cache_path(etf), hedged=True)
        for name, (page, etf) in INDEX_PAGES.items()
    }
    for name, trades in per_index_hedged.items():
        c, taken, _ = simulate(trades, calendar, frac=1.0, start_cash=100_000.0)
        dd = max_drawdown(c)
        print(f"  {name:16} final ${c[-1][1]:>10,.0f} ({c[-1][1]/1e5-1:+7.1%})  "
              f"maxDD {dd['mdd']*100:6.1f}%  n={len(taken)}")
    eqh = simulate_pooled(per_index_hedged, calendar, equal_weights(list(per_index_hedged)))
    dd_eqh = max_drawdown(eqh["combined"])
    print(f"  POOLED equal-weight combined final ${eqh['combined'][-1][1]:>10,.0f} "
          f"({eqh['combined'][-1][1]/1e5-1:+7.1%})  maxDD {dd_eqh['mdd']*100:6.1f}%")

    print("\n===== PROMPT ENTRY (close[AD], captures the overnight gap) =====")
    per_index_prompt = {
        name: build_index_trades(page, etf, wiki_cache=_wiki_cache_path(etf), entry="close_same")
        for name, (page, etf) in INDEX_PAGES.items()
    }
    for name, trades in per_index_prompt.items():
        rets = [t.ret for t in trades]
        c, taken, _ = simulate(trades, calendar, frac=1.0, start_cash=100_000.0)
        print(f"  {name:16} n={len(trades):3}  arith_mean={st.mean(rets)*100:+6.2f}%  "
              f"sd={st.pstdev(rets)*100:5.2f}%  win={sum(r>0 for r in rets)/len(rets)*100:3.0f}%  "
              f"all-in compounded={c[-1][1]/1e5-1:+10.1%}  (illustrative only — see caveat)")

    print("\n===== POSITION-SIZE SENSITIVITY: the fix for both entry conventions =====")
    print("  (frac = fraction of current sleeve equity risked per trade)")
    short = {"S&P 500": "SPY", "S&P 400 MidCap": "MDY", "S&P 600 SmallCap": "IJR", "Nasdaq-100": "QQQ"}
    for label, per_index_x in [("open_next (realistic)", per_index), ("close_same (prompt)", per_index_prompt)]:
        print(f"  -- {label} --")
        for frac in (1.0, 0.5, 0.25, 0.10):
            row = []
            for name, trades in per_index_x.items():
                c, _, _ = simulate(trades, calendar, frac=frac, start_cash=100_000.0)
                row.append(f"{short.get(name, name):>4}={c[-1][1]/1e5-1:+7.0%}")
            print(f"    frac={frac*100:4.0f}%   " + "  ".join(row))


if __name__ == "__main__":
    main()
