"""$100k pocket simulation of the index-inclusion long — concrete P&L & drawdown.

Turns the event study into an actual account: what happens to a real pocket that
trades every S&P 500 addition with *executable* prices and no leverage.

Concrete, unambiguous execution (no slippage — small size):
  ENTRY: BUY at OPEN on AD+1 (first session after the after-close announcement;
         a market-on-open order — no foreknowledge required)
  EXIT : SELL at CLOSE on ED-1 (market-on-close into the reconstitution)

The pocket compounds on **raw** returns (what you actually transact at, not the
market-adjusted alpha spread). Max drawdown is measured on the **daily
mark-to-market** equity curve over the benchmark's full trading calendar (so
flat-cash gaps between trades and unrealized mid-hold dips both count).

The pure functions here (``build_trade``, ``simulate``, ``max_drawdown`` …) take
price frames and are unit-tested with no network.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from .study import locate


@dataclass
class Trade:
    ticker: str
    entry_date: str
    exit_date: str
    entry_px: float
    exit_px: float
    marks: dict = field(repr=False)  # iso_date -> close (entry_date..exit_date)

    @property
    def ret(self) -> float:
        return self.exit_px / self.entry_px - 1.0


def build_trade(df: pd.DataFrame, announcement: dt.date, effective: dt.date, ticker: str):
    """Construct one Trade from a raw OHLC frame (columns ``open``/``close``).

    Returns ``None`` if the window can't be aligned. Entry is the first trading
    day strictly after the announcement; exit is the last trading day strictly
    before the effective date.
    """
    idx = df.index
    ann_i = int(idx.searchsorted(pd.Timestamp(announcement), side="right"))
    eff_i = int(idx.searchsorted(pd.Timestamp(effective), side="left")) - 1
    if ann_i >= len(df) or eff_i < ann_i:
        return None
    entry_px = float(df["open"].iat[ann_i])
    if not entry_px or pd.isna(entry_px):
        return None
    marks = {
        idx[k].date().isoformat(): float(df["close"].iat[k])
        for k in range(ann_i, eff_i + 1)
    }
    exit_date = idx[eff_i].date().isoformat()
    return Trade(ticker, idx[ann_i].date().isoformat(), exit_date,
                 entry_px, marks[exit_date], marks)


def build_trade_prompt(df: pd.DataFrame, announcement: dt.date, effective: dt.date, ticker: str):
    """Like :func:`build_trade`, but entered at ``close[AD]`` (the announcement
    day itself) instead of ``open[AD+1]``.

    This captures the overnight announcement-day gap that :func:`build_trade`'s
    no-foreknowledge ``open[AD+1]`` entry misses by construction — for the S&P
    500 that gap was ~4.5 of the ~6.5pp full move (see FINDINGS.md); for the
    mid/small-cap indices it turns out to be *effectively the entire edge* (the
    ``open[AD+1]`` entry nets ~zero-to-negative there). Trading this requires
    placing the order at (or immediately after) the close on the day S&P
    announces the change after-hours — materially harder to execute than a
    plain next-morning market order, and not achievable at all if you have no
    same-day visibility into the announcement.
    """
    idx = df.index
    ann_i = int(idx.searchsorted(pd.Timestamp(announcement), side="left"))
    eff_i = int(idx.searchsorted(pd.Timestamp(effective), side="left")) - 1
    if ann_i >= len(df) or eff_i < ann_i:
        return None
    entry_px = float(df["close"].iat[ann_i])
    if not entry_px or pd.isna(entry_px):
        return None
    marks = {
        idx[k].date().isoformat(): float(df["close"].iat[k])
        for k in range(ann_i, eff_i + 1)
    }
    exit_date = idx[eff_i].date().isoformat()
    if exit_date not in marks:
        return None
    return Trade(ticker, idx[ann_i].date().isoformat(), exit_date,
                 entry_px, marks[exit_date], marks)


def build_hedged_trade(stock_df: pd.DataFrame, bench_df: pd.DataFrame,
                       announcement: dt.date, effective: dt.date, ticker: str):
    """Like :func:`build_trade`, but long-stock/short-benchmark market-neutral.

    Returns a :class:`Trade` whose price path is a synthetic NAV — ``entry_px``
    is still the stock's real entry price (so position sizing is unaffected),
    but the path/exit track ``stock_return - benchmark_return`` instead of the
    raw stock price. This isolates the announcement→rebalance *alpha* from
    the stock's incidental market-beta exposure over the hold, at the cost of
    financing/borrow cost on the short leg (ignored here — see FINDINGS.md).

    The hedge leg is sized off the benchmark's close on the announcement day
    (matching the stock's open-the-next-session entry closely enough for a
    liquid index ETF); both legs exit together at ``close[ED-1]``.
    """
    idx = stock_df.index.intersection(bench_df.index).sort_values()
    if len(idx) < 5:
        return None
    s_open = stock_df["open"].reindex(idx)
    s_close = stock_df["close"].reindex(idx)
    b_close = bench_df["close"].reindex(idx)
    ann_i = int(idx.searchsorted(pd.Timestamp(announcement), side="right"))
    eff_i = int(idx.searchsorted(pd.Timestamp(effective), side="left")) - 1
    if ann_i >= len(idx) or eff_i < ann_i:
        return None
    s0, b0 = s_open.iat[ann_i], b_close.iat[ann_i]
    if not s0 or not b0 or pd.isna(s0) or pd.isna(b0):
        return None
    s0, b0 = float(s0), float(b0)
    marks = {}
    for k in range(ann_i, eff_i + 1):
        sk, bk = s_close.iat[k], b_close.iat[k]
        if pd.isna(sk) or pd.isna(bk):
            continue
        marks[idx[k].date().isoformat()] = s0 * (float(sk) / s0 - (float(bk) / b0 - 1.0))
    exit_date = idx[eff_i].date().isoformat()
    if exit_date not in marks:
        return None
    return Trade(ticker, idx[ann_i].date().isoformat(), exit_date, s0, marks[exit_date], marks)


def simulate(trades: list[Trade], calendar: list[str], frac: float = 1.0,
             start_cash: float = 100_000.0):
    """Sequential mark-to-market sim over ``calendar`` (list of iso dates).

    Deploys ``frac`` of current equity into each opportunity you're free to fund
    (i.e. flat); the remainder sits in cash at 0%. ``frac=1.0`` is all-in, one
    name at a time. Opportunities that start while a position is open are skipped.
    Returns ``(equity_curve, taken, skipped)``.
    """
    by_entry: dict[str, list[Trade]] = {}
    for t in trades:
        by_entry.setdefault(t.entry_date, []).append(t)

    curve, cash, holding, shares, last_close = [], start_cash, None, 0.0, 0.0
    taken, skipped = [], []
    for day in calendar:
        if day in by_entry:
            for t in by_entry[day]:
                if holding is None:
                    invest = frac * cash
                    shares, cash, holding, last_close = invest / t.entry_px, cash - invest, t, t.entry_px
                    taken.append(t)
                else:
                    skipped.append(t)
        if holding is not None:
            last_close = holding.marks.get(day, last_close)  # carry forward on halts
            equity = cash + shares * last_close
            if day == holding.exit_date:
                cash += shares * holding.exit_px
                equity, holding, shares = cash, None, 0.0
        else:
            equity = cash
        curve.append((day, equity))
    return curve, taken, skipped


def max_drawdown(curve: list[tuple[str, float]]) -> dict:
    """Peak-to-trough stats on an equity curve."""
    peak, peak_date = curve[0][1], curve[0][0]
    mdd, out = 0.0, {"mdd": 0.0, "peak_date": None, "trough_date": None,
                     "peak": curve[0][1], "trough": curve[0][1]}
    for d, e in curve:
        if e > peak:
            peak, peak_date = e, d
        dd = e / peak - 1.0
        if dd < mdd:
            mdd = dd
            out = {"mdd": mdd, "peak_date": peak_date, "trough_date": d,
                   "peak": peak, "trough": e}
    return out


def longest_underwater(curve: list[tuple[str, float]]) -> int:
    """Longest run (in calendar entries) spent below a prior peak."""
    peak, run, longest = curve[0][1], 0, 0
    for _, e in curve:
        if e >= peak:
            peak, run = e, 0
        else:
            run += 1
            longest = max(longest, run)
    return longest


def combine_curves(curves: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    """Sum multiple equity curves that share the same date-aligned calendar.

    Used to aggregate independent per-index "sleeves" (each with its own slice
    of capital) into one portfolio equity curve. All curves must have identical
    dates at each position (build them from the same ``calendar`` argument to
    :func:`simulate`).
    """
    if not curves:
        return []
    n = len(curves[0])
    if any(len(c) != n for c in curves):
        raise ValueError("all curves must share the same calendar length")
    return [
        (curves[0][i][0], sum(c[i][1] for c in curves))
        for i in range(n)
    ]


def worst_losing_streak(taken: list[Trade]) -> tuple[float, list[str]]:
    """Most negative compounded run of consecutive losing trades."""
    worst, cur, worst_run = 0.0, [], []
    for t in taken:
        if t.ret < 0:
            cur.append(t)
            running = 1.0
            for x in cur:
                running *= 1 + x.ret
            if running - 1 < worst:
                worst, worst_run = running - 1, list(cur)
        else:
            cur = []
    return worst, [t.ticker for t in worst_run]
