"""Event-study engine for the S&P 500 "index effect".

Everything here is a pure function over price data (``pandas`` Series/DataFrame)
so it can be unit-tested with no network. The data plumbing lives in
:mod:`index_effect.prices`; the reproducible experiment in
:mod:`index_effect.report`.

Return model
------------
We use **market-adjusted** returns: the stock's simple return over a window
minus the benchmark's (SPY) return over the same window. For the short windows
of an inclusion event (a few trading days), estimating a per-name beta adds far
more variance than the ``beta = 1`` assumption removes, so market-adjustment is
the standard, robust choice.

Because a *small* trader has negligible market impact, the market-adjusted
cumulative abnormal return (CAR) over a tradable window IS the achievable edge,
before (tiny) commissions and borrow.

Event timing convention
------------------------
S&P index additions are effective at the **open** of the effective date (ED);
index funds execute the reconstitution trade at the **close of the prior
trading day** (ED-1). Forced buying therefore peaks into ``close[ED-1]``. S&P
announces after the market close on the announcement date (AD), so the market's
first opportunity to react is the ``AD -> AD+1`` session.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import pandas as pd


def align(stock: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """Inner-join stock and benchmark closes on their common trading days."""
    df = pd.DataFrame({"stock": stock, "bench": bench}).dropna()
    return df.sort_index()


def locate(index: pd.DatetimeIndex, when: dt.date) -> int:
    """Positional index of the first trading day on/after ``when``."""
    ts = pd.Timestamp(when)
    return int(index.searchsorted(ts, side="left"))


def car(df: pd.DataFrame, i: int, j: int) -> float:
    """Market-adjusted cumulative abnormal return from ``close[i]`` to ``close[j]``.

    Uses the ``stock`` and ``bench`` columns produced by :func:`align`.
    """
    s = df["stock"].iat[j] / df["stock"].iat[i] - 1.0
    m = df["bench"].iat[j] / df["bench"].iat[i] - 1.0
    return float(s - m)


def has_split_artifact(df: pd.DataFrame, lo: int, hi: int, thresh: float = 0.40) -> bool:
    """True if any single-day raw move in ``[lo, hi)`` exceeds ``thresh`` (likely
    an unadjusted split)."""
    s = df["stock"].to_numpy()
    for k in range(max(0, lo), min(hi, len(s) - 1)):
        if abs(s[k + 1] / s[k] - 1.0) > thresh:
            return True
    return False


# Windows expressed as (name, anchor, start_offset, end_offset). ``anchor`` is
# "ED" (effective date) or "AD" (announcement date); offsets are in trading days
# relative to the anchor's positional index.
ED_WINDOWS = {
    "runup_ED6_ED1": ("ED", -6, -1),  # buy close[ED-6], sell into rebalance close[ED-1]
    "runup_ED10_ED1": ("ED", -10, -1),
    "reversal_ED1_ED5": ("ED", -1, 5),  # long P&L; SHORT this to trade the reversal
    "reversal_ED1_ED10": ("ED", -1, 10),
    "reversal_ED1_ED20": ("ED", -1, 20),
}
AD_WINDOWS = {
    "ann_day": ("AD", -1, 0),  # close[AD-1] -> close[AD]  (news lands after close)
    "buy_ADclose_ED1": ("AD_ED", 0, -1),  # close[AD] -> close[ED-1]
    "buy_ADclose_ED": ("AD_ED", 0, 0),  # close[AD] -> close[ED]
}


@dataclass
class EventResult:
    ticker: str
    effective: dt.date
    announcement: dt.date | None
    dollar_volume: float  # pre-event median daily $volume
    cars: dict  # window name -> CAR (float) or None


def study_event(
    df: pd.DataFrame,
    effective: dt.date,
    announcement: dt.date | None,
    ticker: str,
    volume: pd.Series | None = None,
    pre_lo: int = -25,
    pre_hi: int = -10,
    pad: int = 22,
) -> EventResult | None:
    """Compute the standard CAR battery for one addition event.

    Returns ``None`` if the aligned data lacks enough room around the event or a
    split artifact is detected. ``volume`` (share volume, same index as ``df``)
    drives the pre-event liquidity estimate used for the liquidity split.
    """
    idx = df.index
    iE = locate(idx, effective)
    if iE < -pre_lo or iE + pad >= len(df):
        return None
    if has_split_artifact(df, iE + pre_lo, iE + pad):
        return None

    dollar_volume = float("nan")
    if volume is not None:
        v = volume.reindex(idx)
        seg = (df["stock"] * v).iloc[iE + pre_lo : iE + pre_hi].dropna()
        if len(seg):
            dollar_volume = float(seg.median())

    cars: dict = {}
    for name, (anchor, a, b) in ED_WINDOWS.items():
        cars[name] = car(df, iE + a, iE + b)

    if announcement is not None:
        iA = locate(idx, announcement)
        if 0 < iA < iE and iA + 1 < len(df):
            for name, (anchor, a, b) in AD_WINDOWS.items():
                if anchor == "AD":
                    cars[name] = car(df, iA + a, iA + b)
                else:  # AD_ED: start relative to AD, end relative to ED
                    cars[name] = car(df, iA + a, iE + b)
            # buy at the NEXT open (public-info, no-foreknowledge entry) needs the
            # open series; report.py fills "buy_open1_ED1" when opens are supplied.
            cars["lag_trading_days"] = iE - iA

    return EventResult(ticker, effective, announcement, dollar_volume, cars)


def summarize(values) -> dict | None:
    """Sample stats for a list of CARs: n, mean, median, one-sample t, win rate."""
    xs = [
        float(x)
        for x in values
        if x is not None and not (isinstance(x, float) and math.isnan(x))
    ]
    if not xs:
        return None
    n = len(xs)
    mean = sum(xs) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        se = math.sqrt(var / n)
        t = mean / se if se > 0 else float("nan")
    else:
        t = float("nan")
    srt = sorted(xs)
    median = (
        srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2.0
    )
    win = sum(1 for x in xs if x > 0) / n
    return {"n": n, "mean": mean, "median": median, "t": t, "win": win}


def liquidity_terciles(results: list[EventResult]) -> list[tuple[str, list[EventResult]]]:
    """Split events into LOW/MID/HIGH terciles by pre-event median $volume."""
    ok = [r for r in results if not math.isnan(r.dollar_volume)]
    ok.sort(key=lambda r: r.dollar_volume)
    n = len(ok)
    third = n // 3
    return [
        ("LOW $vol", ok[:third]),
        ("MID $vol", ok[third : 2 * third]),
        ("HIGH $vol", ok[2 * third :]),
    ]
