"""Screen a universe of instruments for fit with the Markov regime method.

The method is a regime-persistence / trend-following overlay: its edge comes
from a diagonal-dominant transition matrix (high stickiness) and separable,
durable bull/bear regimes it can ride or sidestep. This module curates two
universes, runs the full model on each name, and ranks them by how well the
strategy actually performs (risk-adjusted), not by raw return.

It reports BOTH the long-short backtest (the default — net short in bear
regimes) and a long-only / de-risk-only baseline, so the "shorting alpha"
(ls_sharpe - lo_sharpe) shows how much going short actually contributes.

Run with::

    python -m quant.screen                      # equity/ETF universe, 10y
    python -m quant.screen --universe macro      # FX / rates / ags / futures
    python -m quant.screen --period 5y
    python -m quant.screen JPY=X ZN=F ZC=F        # custom tickers
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from .regime import MarkovRegimeModel

# ---------------------------------------------------------------------------
# Universe 1 — liquid equity / commodity ETFs (the original screen).
# ---------------------------------------------------------------------------
ETF_UNIVERSE: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "EEM": "Emerging markets",
    "EWZ": "Brazil (EM single-country)",
    "EWJ": "Japan",
    "GLD": "Gold",
    "SLV": "Silver",
    "GDX": "Gold miners",
    "USO": "Crude oil",
    "DBC": "Broad commodities",
    "TLT": "20y+ Treasuries",
    "HYG": "High-yield credit",
    "UUP": "US dollar index",
    "FXE": "Euro",
    "FXY": "Japanese yen",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "XLE": "Energy sector",
    "ARKK": "Disruptive innovation",
    "SMH": "Semiconductors",
}

# ---------------------------------------------------------------------------
# Universe 2 — trending-but-not-secular-bull macro instruments. These are the
# classic CTA / managed-futures markets: FX crosses, rate futures, ags, and
# energy/metals futures. None has the relentless equity drift that lets
# buy & hold dominate, so the regime overlay (especially its shorting leg)
# has room to add value.
# ---------------------------------------------------------------------------
MACRO_UNIVERSE: dict[str, str] = {
    # FX (yfinance "=X"; pairs quoted USD-base unless noted)
    "JPY=X": "USD/JPY",
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "CAD=X": "USD/CAD",
    "MXN=X": "USD/MXN (EM FX)",
    # Rate futures
    "ZN=F": "10y T-note future",
    "ZB=F": "30y T-bond future",
    "ZF=F": "5y T-note future",
    # Agricultural futures
    "ZC=F": "Corn",
    "ZW=F": "Wheat",
    "ZS=F": "Soybeans",
    "KC=F": "Coffee",
    "SB=F": "Sugar",
    "CT=F": "Cotton",
    # Energy / metals futures
    "CL=F": "WTI crude future",
    "NG=F": "Natural gas future",
    "HG=F": "Copper",
    "PL=F": "Platinum",
    "SI=F": "Silver future",
}

UNIVERSES = {"etf": ETF_UNIVERSE, "macro": MACRO_UNIVERSE}


def _bh_sharpe(prices) -> float:
    rets = prices.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return float("nan")
    return float(np.sqrt(252) * rets.mean() / rets.std())


def screen(tickers: dict[str, str], period: str = "10y", window: int = 20) -> list[dict]:
    from .data import load_prices

    rows: list[dict] = []
    for ticker, name in tickers.items():
        rec: dict = {"ticker": ticker, "name": name}
        try:
            prices = load_prices(ticker, period=period)
            model = MarkovRegimeModel(ticker=ticker, window=window)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Long-short (default) and long-only (de-risk-only) baselines.
                ls = model.analyze(
                    prices=prices, run_backtest=True, run_hmm=False, allow_short=True
                )
                lo = model.analyze(
                    prices=prices, run_backtest=True, run_hmm=False, allow_short=False
                )
        except Exception as exc:  # noqa: BLE001 — screening should never abort the batch
            rec["error"] = str(exc)[:60]
            rows.append(rec)
            continue

        ls_bt, lo_bt = ls.backtest, lo.backtest
        rec.update(
            {
                "days": ls.num_labeled_days,
                "max_stick": float(ls.stickiness.max()),
                "state": ls.current_state_name,
                "ls_sharpe": ls_bt.sharpe,
                "lo_sharpe": lo_bt.sharpe,
                "short_alpha": ls_bt.sharpe - lo_bt.sharpe,
                "ls_ret": ls_bt.total_return,
                "max_dd": ls_bt.max_drawdown,
                "bh_ret": ls_bt.buy_hold_return,
                "bh_sharpe": _bh_sharpe(prices),
            }
        )
        rec["excess_sharpe"] = ls_bt.sharpe - rec["bh_sharpe"]
        rows.append(rec)
    return rows


def format_table(rows: list[dict], title: str) -> str:
    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    ok.sort(key=lambda r: (r["ls_sharpe"], r["max_dd"]), reverse=True)

    out: list[str] = []
    out.append("=" * 110)
    out.append(f"  {title} (ranked by long-short Sharpe)")
    out.append("=" * 110)
    out.append(
        f"{'#':>2}  {'ticker':<9}{'name':<22}{'days':>5}{'stick':>7}"
        f"{'LS Shp':>8}{'LO Shp':>8}{'shrtA':>7}{'maxDD':>7}"
        f"{'LSret':>8}{'B&Hret':>9}{'exShp':>7}  {'today':<9}"
    )
    out.append("-" * 110)
    for i, r in enumerate(ok, 1):
        out.append(
            f"{i:>2}  {r['ticker']:<9}{r['name'][:21]:<22}{r['days']:>5}"
            f"{r['max_stick'] * 100:>6.0f}%{r['ls_sharpe']:>8.2f}{r['lo_sharpe']:>8.2f}"
            f"{r['short_alpha']:>+7.2f}{r['max_dd'] * 100:>6.0f}%"
            f"{r['ls_ret'] * 100:>7.0f}%{r['bh_ret'] * 100:>8.0f}%"
            f"{r['excess_sharpe']:>+7.2f}  {r['state']:<9}"
        )
    if bad:
        out.append("-" * 110)
        for r in bad:
            out.append(f"    {r['ticker']:<9}{r['name'][:21]:<22}ERROR: {r['error']}")
    out.append("=" * 110)
    out.append(
        "LS=long-short  LO=long-only(de-risk)  shrtA=shorting alpha (LS-LO Sharpe)  "
        "exShp=LS Sharpe - buy&hold Sharpe"
    )
    out.append("NOT financial advice — educational reproduction of a YouTube method.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quant.screen", description=__doc__)
    p.add_argument("tickers", nargs="*", help="custom tickers (default: chosen universe)")
    p.add_argument("--universe", choices=sorted(UNIVERSES), default="etf")
    p.add_argument("--period", default="10y")
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args(argv)

    if args.tickers:
        universe, title = {t: t for t in args.tickers}, "CUSTOM UNIVERSE SCREEN"
    else:
        universe = UNIVERSES[args.universe]
        title = f"MARKOV REGIME SCREEN — {args.universe.upper()} UNIVERSE"
    rows = screen(universe, period=args.period, window=args.window)
    print(format_table(rows, title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
