"""Network-free tests for the index-effect event study.

Prices are synthetic: a flat benchmark plus a stock with a deliberately injected
inclusion "pop", so the expected CARs are known in closed form.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quant.index_effect.europe import INDEX_CONFIG, NAME_TO_TICKER, extract_additions, parse_change_blocks
from quant.index_effect.events import parse_changes
from quant.index_effect.multi_index import (
    dsr_weights,
    equal_weights,
    simulate_pooled,
    sleeve_correlation,
)
from quant.index_effect.portfolio import (
    Trade,
    build_hedged_trade,
    build_trade,
    build_trade_prompt,
    combine_curves,
    longest_underwater,
    max_drawdown,
    simulate,
    worst_losing_streak,
)
from quant.index_effect.significance import assess, per_trade_sharpe
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
    assert panw.removed == ""  # this fixture row has no removed ticker


def test_parse_changes_captures_removed_ticker():
    wt = NASDAQ_FIXTURE  # ALAB added, WBA removed
    e = parse_changes(wt)[0]
    assert e.ticker == "ALAB" and e.removed == "WBA"


def test_parse_changes_discards_out_of_window_announcement():
    # announcement more than 45 days before effective is treated as untrusted
    wt = WIKITEXT_FIXTURE.replace("June 2, 2023", "January 2, 2023")
    panw = {e.ticker: e for e in parse_changes(wt)}["PANW"]
    assert panw.announcement is None


# Nasdaq/Dow layout: id="changes" table, one cell per line, ISO citation dates.
NASDAQ_FIXTURE = """
{| class="wikitable sortable" id="changes"
! rowspan="2" |Date
! colspan="2" |Added
! colspan="2" |Removed
! rowspan="2" |Reason
|-
!Ticker
!Security
!Ticker
!Security
|-
|June 24, 2024
|ALAB
|[[Astera Labs]]
|WBA
|[[Walgreens]]
|Quarterly reconstitution.<ref>{{Cite web |date=2024-06-07 |title=x |url=http://x}}</ref>
|}
"""


def test_parse_changes_handles_line_per_cell_and_iso_dates():
    events = parse_changes(NASDAQ_FIXTURE)
    assert len(events) == 1
    e = events[0]
    assert e.ticker == "ALAB"
    assert e.effective == dt.date(2024, 6, 24)
    assert e.announcement == dt.date(2024, 6, 7)  # ISO date in citation


def _ohlc(rows, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=idx, columns=["open", "close"])


def test_build_trade_entry_open_exit_close():
    # 10 business days; announce on day-2 date, effective on day-8 date.
    rows = [(100, 100)] * 10
    rows[3] = (110, 112)  # AD+1 open we should buy at (announcement was day index 2)
    rows[6] = (120, 121)  # ED-1 close we should sell at (effective is day index 7)
    df = _ohlc(rows)
    ann = df.index[2].date()
    eff = df.index[7].date()
    t = build_trade(df, ann, eff, "T")
    assert t.entry_px == 110  # OPEN on the first day after AD
    assert t.exit_px == 121  # CLOSE on the last day before ED
    assert abs(t.ret - (121 / 110 - 1)) < 1e-12


def test_build_trade_prompt_enters_at_announcement_day_close():
    # same fixture as test_build_trade_entry_open_exit_close: announce day-2,
    # effective day-7. The prompt entry buys CLOSE on day-2 itself, not the
    # OPEN on day-3 -- capturing the overnight gap between them.
    rows = [(100, 100)] * 10
    rows[2] = (105, 108)  # announcement day's close: what the prompt entry buys
    rows[3] = (110, 112)  # AD+1 open/close: what the no-foreknowledge entry buys
    rows[6] = (120, 121)
    df = _ohlc(rows)
    ann, eff = df.index[2].date(), df.index[7].date()
    t = build_trade_prompt(df, ann, eff, "T")
    assert t.entry_px == 108  # CLOSE on the announcement day
    assert t.exit_px == 121
    assert abs(t.ret - (121 / 108 - 1)) < 1e-12
    # strictly more return captured than the no-foreknowledge open[AD+1] entry
    open_next = build_trade(df, ann, eff, "T")
    assert t.ret > open_next.ret


def test_build_hedged_trade_cancels_matched_beta():
    # stock +10% and benchmark +10% over the same hold -> zero hedged return
    stock = [(100, 100)] * 10
    stock[3] = (110, 112)
    stock[6] = (120, 121)  # stock: 110 -> 121, +10%
    bench = [(100, 100)] * 10
    bench[6] = (100, 110)  # bench close[3]=100 (anchor) -> close[6]=110, +10%
    sdf, bdf = _ohlc(stock), _ohlc(bench)
    ann, eff = sdf.index[2].date(), sdf.index[7].date()
    t = build_hedged_trade(sdf, bdf, ann, eff, "T")
    assert t is not None
    assert abs(t.ret - 0.0) < 1e-9


def test_build_hedged_trade_isolates_alpha_from_beta():
    # stock +10%, benchmark flat -> hedged return = the full +10% (no beta to strip)
    stock = [(100, 100)] * 10
    stock[3] = (110, 112)
    stock[6] = (120, 121)
    bench = [(100, 100)] * 10  # perfectly flat
    sdf, bdf = _ohlc(stock), _ohlc(bench)
    ann, eff = sdf.index[2].date(), sdf.index[7].date()
    t = build_hedged_trade(sdf, bdf, ann, eff, "T")
    assert abs(t.ret - (121 / 110 - 1)) < 1e-9


def test_build_hedged_trade_short_leg_profits_when_benchmark_falls():
    # stock flat, benchmark falls 10% -> hedged return is positive (short gains)
    stock = [(100, 100)] * 10  # flat: open[3]=100, close[6]=100
    bench = [(100, 100)] * 10
    bench[6] = (100, 90)  # bench close[3]=100 (anchor) -> close[6]=90, -10%
    sdf, bdf = _ohlc(stock), _ohlc(bench)
    ann, eff = sdf.index[2].date(), sdf.index[7].date()
    t = build_hedged_trade(sdf, bdf, ann, eff, "T")
    assert abs(t.ret - 0.10) < 1e-9


def test_simulate_compounds_and_respects_no_leverage():
    df = _ohlc([(100, 100)] * 12)
    # two sequential winners of +10% each (open 100 -> close 110)
    for i in (2, 8):
        df.iloc[i, 0] = 100.0  # open
    # trade 1: buy open[1]=100, sell close[3]
    df.iloc[1, 0] = 100.0
    df.iloc[3, 1] = 110.0
    df.iloc[7, 0] = 100.0
    df.iloc[9, 1] = 110.0
    t1 = build_trade(df, df.index[0].date(), df.index[4].date(), "A")
    t2 = build_trade(df, df.index[6].date(), df.index[10].date(), "B")
    cal = [d.date().isoformat() for d in df.index]
    curve, taken, skipped = simulate([t1, t2], cal, frac=1.0, start_cash=100.0)
    assert len(taken) == 2 and not skipped
    # 100 * 1.10 * 1.10 = 121
    assert abs(curve[-1][1] - 121.0) < 1e-6


def test_simulate_skips_overlapping_and_fraction_scales_drawdown():
    df = _ohlc([(100, 100)] * 8)
    df.iloc[1, 0] = 100.0
    df.iloc[4, 1] = 80.0  # a -20% loser
    loser = build_trade(df, df.index[0].date(), df.index[5].date(), "L")
    cal = [d.date().isoformat() for d in df.index]
    full = max_drawdown(simulate([loser], cal, frac=1.0, start_cash=100.0)[0])["mdd"]
    half = max_drawdown(simulate([loser], cal, frac=0.5, start_cash=100.0)[0])["mdd"]
    assert full < -0.15  # ~ -20%
    assert half > full  # half size => shallower drawdown
    assert abs(half - full / 2) < 0.02  # ~ linear in position size


def test_max_drawdown_and_underwater():
    curve = [("d1", 100), ("d2", 120), ("d3", 90), ("d4", 110), ("d5", 108)]
    dd = max_drawdown(curve)
    assert abs(dd["mdd"] - (90 / 120 - 1)) < 1e-12
    assert dd["peak_date"] == "d2" and dd["trough_date"] == "d3"
    assert longest_underwater(curve) == 3  # d3,d4,d5 all below the 120 peak


def test_worst_losing_streak():
    class T:
        def __init__(self, r):
            self.ret = r
            self.ticker = f"{r:+.2f}"
    trades = [T(0.05), T(-0.10), T(-0.05), T(0.03), T(-0.20)]
    worst, names = worst_losing_streak(trades)
    # candidates: [-10%,-5%] => -14.5% ; lone [-20%] => -20% (the worst)
    assert abs(worst - (-0.20)) < 1e-9
    assert names == ["-0.20"]


def test_per_trade_sharpe():
    assert per_trade_sharpe([0.01, 0.01, 0.01]) == 0.0  # zero variance -> guarded to 0
    assert per_trade_sharpe([]) == 0.0
    # positive mean with spread -> positive finite Sharpe
    s = per_trade_sharpe([0.05, -0.01, 0.03, 0.02])
    assert s > 0 and s < 10


def test_assess_ranks_clean_edge_above_noise():
    # a consistently positive edge should earn higher PSR/DSR than a noisy one
    strong = [0.04, 0.05, 0.03, 0.06, 0.04, 0.05, 0.03, 0.05]
    weak = [0.05, -0.06, 0.07, -0.04, 0.06, -0.05, 0.04, -0.03]
    out = assess({"strong": strong, "weak": weak}, n_trials=10)
    assert out["strong"]["psr"] > out["weak"]["psr"]
    assert out["strong"]["dsr"] > out["weak"]["dsr"]
    assert 0.0 <= out["strong"]["dsr"] <= 1.0


def test_combine_curves_sums_aligned_equity():
    a = [("d1", 100.0), ("d2", 110.0), ("d3", 90.0)]
    b = [("d1", 50.0), ("d2", 50.0), ("d3", 60.0)]
    combined = combine_curves([a, b])
    assert combined == [("d1", 150.0), ("d2", 160.0), ("d3", 150.0)]


def test_combine_curves_rejects_misaligned_lengths():
    a = [("d1", 100.0), ("d2", 110.0)]
    b = [("d1", 50.0)]
    with pytest.raises(ValueError):
        combine_curves([a, b])


def test_combine_curves_empty():
    assert combine_curves([]) == []


def _mk_trade(entry, exit_, entry_px, exit_px):
    return Trade("X", entry, exit_, entry_px, exit_px, {entry: entry_px, exit_: exit_px})


def test_equal_weights_sum_to_one():
    w = equal_weights(["A", "B", "C", "D"])
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(abs(v - 0.25) < 1e-9 for v in w.values())


def test_dsr_weights_favor_the_cleaner_edge_and_sum_to_one():
    # index A: consistent small winner; index B: noisy coin-flip -> A should
    # get more capital, and a near-zero-edge sleeve still gets the floor, not 0.
    strong = [_mk_trade(f"d{i}", f"d{i}x", 100, 100 * (1 + r))
              for i, r in enumerate([0.02, 0.03, 0.025, 0.02, 0.03, 0.025, 0.02, 0.03])]
    weak = [_mk_trade(f"e{i}", f"e{i}x", 100, 100 * (1 + r))
            for i, r in enumerate([0.05, -0.06, 0.07, -0.04, 0.06, -0.05, 0.04, -0.03])]
    w = dsr_weights({"strong": strong, "weak": weak})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["strong"] > w["weak"]
    assert w["weak"] > 0  # floor keeps a token allocation, never zero


def test_dsr_weights_handles_empty_sleeve():
    strong = [_mk_trade(f"d{i}", f"d{i}x", 100, 102) for i in range(5)]
    w = dsr_weights({"strong": strong, "empty": []})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["empty"] > 0  # floor, not a crash


def test_simulate_pooled_matches_manual_sleeve_sum():
    t1 = _mk_trade("d0", "d1", 100, 110)  # sleeve A: +10%
    t2 = _mk_trade("d0", "d1", 100, 90)  # sleeve B: -10%
    cal = ["d0", "d1"]
    weights = {"A": 0.5, "B": 0.5}
    out = simulate_pooled({"A": [t1], "B": [t2]}, cal, weights, start_cash=100.0)
    # 50 into a +10% winner and 50 into a -10% loser -> net back to par
    assert abs(out["combined"][-1][1] - 100.0) < 1e-9
    assert abs(out["sleeves"]["A"][-1][1] - 55.0) < 1e-9
    assert abs(out["sleeves"]["B"][-1][1] - 45.0) < 1e-9


def test_sleeve_correlation_perfectly_correlated_sleeves():
    # varying daily returns, but b always moves at 2x a's return -> corr = 1
    a = [("d0", 100.0), ("d1", 110.0), ("d2", 99.0), ("d3", 108.9)]
    b = [("d0", 100.0), ("d1", 120.0), ("d2", 96.0), ("d3", 115.2)]
    corr = sleeve_correlation({"a": a, "b": b})
    assert abs(corr["a / b"] - 1.0) < 1e-9


def test_sleeve_correlation_uncorrelated_sleeves():
    a = [("d0", 100.0), ("d1", 110.0), ("d2", 99.0), ("d3", 108.9)]
    b = [("d0", 100.0), ("d1", 99.0), ("d2", 108.9), ("d3", 98.0)]
    corr = sleeve_correlation({"a": a, "b": b})
    assert -1.0 <= corr["a / b"] <= 1.0


# ---- europe.py: STOXX historical-compositions PDF-text parser ----

STOXX_TEXT_SIMPLE = """
 HISTORICAL INDEX COMPOSITIONS OF EQUITY AND STRATEGY INDICES


4/60
Date of
change
Date of
announcement Deletion Addition
23.07.2001 26.06.2001 Dresdner Bank MLP
24.09.2018 05.09.2018 Commerzbank AG Wirecard AG
"""

STOXX_TEXT_MULTI_ADD = """
20.09.2021 03.09.2021 -
Airbus SE
Brenntag SE
HelloFresh SE
"""

STOXX_TEXT_AMBIGUOUS = """
24.06.2024 18.06.2024 MorphoSys Elmos Semiconductor
23.12.2024 04.12.2024 Energiekontor
SMA Solar Technology
"""


def test_parse_change_blocks_basic():
    blocks = parse_change_blocks(STOXX_TEXT_SIMPLE, min_year=2015)
    # the 2001 row is filtered out by min_year; only 2018 survives
    assert len(blocks) == 1
    eff, ann, pairs = blocks[0]
    assert eff == dt.date(2018, 9, 24)
    assert ann == dt.date(2018, 9, 5)
    assert pairs == ["Commerzbank AG Wirecard AG"]


def test_parse_change_blocks_multiline_pure_addition():
    blocks = parse_change_blocks(STOXX_TEXT_MULTI_ADD, min_year=2015)
    assert len(blocks) == 1
    eff, ann, pairs = blocks[0]
    assert eff == dt.date(2021, 9, 20)
    assert pairs == ["-", "Airbus SE", "Brenntag SE", "HelloFresh SE"]


def test_extract_additions_resolves_simple_swap():
    names = {"Commerzbank AG": "CBK.DE", "Wirecard AG": "WDI.HM"}
    additions, dropped = extract_additions(STOXX_TEXT_SIMPLE, min_year=2015, name_to_ticker=names)
    assert additions == [(dt.date(2018, 9, 24), dt.date(2018, 9, 5), "WDI.HM")]
    assert dropped == []


def test_extract_additions_resolves_pure_addition_block():
    names = {"Airbus SE": "AIR.DE", "Brenntag SE": "BNR.DE", "HelloFresh SE": "HFG.DE"}
    additions, dropped = extract_additions(STOXX_TEXT_MULTI_ADD, min_year=2015, name_to_ticker=names)
    tickers = {t for _, _, t in additions}
    assert tickers == {"AIR.DE", "BNR.DE", "HFG.DE"}
    assert all(eff == dt.date(2021, 9, 20) for eff, _, _ in additions)
    assert dropped == []


def test_index_config_covers_the_full_german_market_cap_ladder():
    assert set(INDEX_CONFIG) == {"DAX", "TecDAX", "MDAX", "SDAX"}
    for marker, bench in INDEX_CONFIG.values():
        assert "INDEX COMPOSITION" in marker
        assert bench.startswith("^")


def test_name_to_ticker_has_no_blank_entries():
    for name, ticker in NAME_TO_TICKER.items():
        assert name.strip() and ticker.strip()
        assert "." in ticker  # every entry is an exchange-suffixed ticker


def test_extract_additions_drops_ambiguous_blocks_instead_of_guessing():
    # only Elmos Semiconductor is a known name in the dict; MorphoSys, Energiekontor
    # and SMA Solar Technology are deliberately left out to exercise the drop path.
    names = {"Elmos Semiconductor": "ELG.DE"}
    additions, dropped = extract_additions(STOXX_TEXT_AMBIGUOUS, min_year=2015, name_to_ticker=names)
    assert additions == []  # 1-known-name lines can't be resolved to a addition/deletion pair
    assert len(dropped) == 3
    assert all(reason for *_, reason in dropped)
