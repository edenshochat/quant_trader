"""Network-free tests for the index-effect event study.

Prices are synthetic: a flat benchmark plus a stock with a deliberately injected
inclusion "pop", so the expected CARs are known in closed form.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from quant.index_effect.events import parse_changes
from quant.index_effect.study import (
    align,
    car,
    has_split_artifact,
    liquidity_terciles,
    locate,
    study_event,
    summarize,
)


def _frame(closes, start="2024-01-01", bench_level=100.0):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    stock = pd.Series(closes, index=idx, name="stock")
    bench = pd.Series(np.full(len(closes), bench_level), index=idx, name="bench")
    return pd.DataFrame({"stock": stock, "bench": bench})


def test_car_market_adjusts():
    df = _frame([100.0, 110.0, 121.0])
    # flat benchmark, so CAR equals the raw stock return
    assert abs(car(df, 0, 1) - 0.10) < 1e-12
    assert abs(car(df, 0, 2) - 0.21) < 1e-12


def test_car_subtracts_benchmark():
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    df = pd.DataFrame(
        {"stock": pd.Series([100.0, 110.0], index=idx),
         "bench": pd.Series([100.0, 104.0], index=idx)}
    )
    assert abs(car(df, 0, 1) - (0.10 - 0.04)) < 1e-12


def test_locate_finds_first_on_or_after():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({"stock": range(10), "bench": range(10)}, index=idx)
    assert locate(df.index, dt.date(2024, 1, 1)) == 0
    # a weekend date resolves to the next business day's position
    assert df.index[locate(df.index, dt.date(2024, 1, 6))] >= pd.Timestamp("2024-01-06")


def test_study_event_recovers_injected_pop():
    # 90 flat days, +8% run-up over the 5 days into ED-1, then a -3% reversal.
    n = 90
    closes = [100.0] * n
    eff_pos = 45  # ED index (leaves room for the -25..+22 windows)
    for k in range(eff_pos - 5, eff_pos):  # buy ED-6..sell ED-1 spans the run-up
        closes[k] = closes[k - 1] * (1 + 0.08 / 5)
    plateau = closes[eff_pos - 2]
    for k in range(eff_pos - 1, n):
        closes[k] = plateau
    # linear give-back reaching -3% at ED+20 (inclusive), then hold
    for k in range(eff_pos - 1, eff_pos + 21):
        closes[k] = plateau * (1 - 0.03 * (k - (eff_pos - 1)) / 21)
    for k in range(eff_pos + 21, n):
        closes[k] = closes[eff_pos + 20]
    df = _frame(closes)
    res = study_event(df, df.index[eff_pos].date(), None, "TEST",
                      volume=pd.Series(1e6, index=df.index))
    assert res is not None
    assert res.cars["runup_ED6_ED1"] > 0.05  # captured most of the +8%
    assert res.cars["reversal_ED1_ED20"] < 0  # reversal is negative (short it)


def test_study_event_rejects_split_artifact():
    closes = [100.0] * 60
    closes[41] = 50.0  # 2:1 split not adjusted -> huge fake move near ED
    df = _frame(closes)
    assert study_event(df, df.index[40].date(), None, "SPLIT") is None


def test_has_split_artifact():
    df = _frame([100, 101, 205, 206])
    assert has_split_artifact(df, 0, 4)
    df2 = _frame([100, 101, 102, 103])
    assert not has_split_artifact(df2, 0, 4)


def test_summarize_stats():
    s = summarize([0.01, 0.02, 0.03, 0.04])
    assert s["n"] == 4
    assert abs(s["mean"] - 0.025) < 1e-12
    assert s["win"] == 1.0
    assert s["t"] > 0
    assert summarize([]) is None
    assert summarize([None, float("nan")]) is None


def test_liquidity_terciles_orders_low_to_high():
    class R:
        def __init__(self, dv):
            self.dollar_volume = dv
    rs = [R(v) for v in [500, 100, 300, 50, 400, 200, 600, 150, 350]]
    buckets = liquidity_terciles(rs)
    names = [n for n, _ in buckets]
    assert names == ["LOW $vol", "MID $vol", "HIGH $vol"]
    low = [r.dollar_volume for r in buckets[0][1]]
    high = [r.dollar_volume for r in buckets[2][1]]
    assert max(low) <= min(high)


WIKITEXT_FIXTURE = """
==Selected changes to the list of S&P 500 components==
{|
|-
! Effective Date !! Ticker || Security || Ticker || Security || Reason
|-
|June 20, 2023 || PANW || [[Palo Alto Networks]] ||  ||  || M&A.<ref name="a">{{cite web |url=http://x |title=t |publisher=S&P |date=June 2, 2023}}</ref>
|-
|September 20, 2021 || MTCH || [[Match Group]] ||  ||  || Change.<ref>{{cite web |date=September 3, 2021 |title=t}}</ref>
|-
|March 15, 2024 ||  ||  || XYZ || [[Removed Co]] || Only a removal.
|}
"""


def test_parse_changes_extracts_additions_and_announcement():
    events = parse_changes(WIKITEXT_FIXTURE)
    tickers = {e.ticker: e for e in events}
    assert "PANW" in tickers and "MTCH" in tickers
    assert "XYZ" not in tickers  # removal-only row is not an addition
    panw = tickers["PANW"]
    assert panw.effective == dt.date(2023, 6, 20)
    assert panw.announcement == dt.date(2023, 6, 2)
    assert panw.lag_calendar_days == 18


def test_parse_changes_discards_out_of_window_announcement():
    # announcement more than 45 days before effective is treated as untrusted
    wt = WIKITEXT_FIXTURE.replace("June 2, 2023", "January 2, 2023")
    panw = {e.ticker: e for e in parse_changes(wt)}["PANW"]
    assert panw.announcement is None
