"""Profit mode: Monte-Carlo "what would $X have made?" over random months.

Given a ticker, a date window (default: last 5 years), a money amount (default
$1000) and a number of months to sample (default 12), this repeatedly draws a
random set of calendar months from the window, compounds the amount through the
strategy's realized return in each sampled month, and reports the distribution
of outcomes — alongside the same draw applied to buy & hold.

The strategy returns come from the same no-look-ahead walk-forward backtest the
rest of the package uses, grouped into monthly buckets. Sampling is without
replacement when the window has enough months, otherwise it bootstraps (samples
with replacement) and says so.

Run with::

    python -m quant.profit SPY                       # 12 months, last 5y, $1000
    python -m quant.profit GLD --months 6 --amount 5000
    python -m quant.profit BTC-USD --start 2021-01 --end 2024-12 --trials 5000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import walk_forward
from .data import load_prices
from .states import DEFAULT_BEAR_THRESHOLD, DEFAULT_BULL_THRESHOLD, DEFAULT_WINDOW


def monthly_returns(
    prices: pd.Series,
    window: int = DEFAULT_WINDOW,
    bull_threshold: float = DEFAULT_BULL_THRESHOLD,
    bear_threshold: float = DEFAULT_BEAR_THRESHOLD,
    leverage: float = 1.0,
    max_exposure: float = 1.0,
    allow_short: bool = True,
    vol_target: float | None = None,
    max_leverage: float = 3.0,
    jump_penalty: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Monthly compounded returns for (strategy, buy & hold), indexed by month.

    Strategy returns come from the walk-forward backtest; buy & hold uses the
    same traded days so the two are directly comparable.
    """
    bt = walk_forward(
        prices,
        window,
        bull_threshold,
        bear_threshold,
        leverage=leverage,
        max_exposure=max_exposure,
        allow_short=allow_short,
        vol_target=vol_target,
        max_leverage=max_leverage,
        jump_penalty=jump_penalty,
    )
    daily_strat = bt.strategy_returns
    daily_bh = prices.pct_change().reindex(daily_strat.index)

    by_month = daily_strat.index.to_period("M")
    strat_m = (1.0 + daily_strat).groupby(by_month).prod() - 1.0
    bh_m = (1.0 + daily_bh).groupby(by_month).prod() - 1.0
    return strat_m, bh_m


@dataclass
class ProfitStats:
    median: float
    mean: float
    p5: float
    p95: float
    worst: float
    best: float
    prob_profit: float


def _stats(final_values: np.ndarray, amount: float) -> ProfitStats:
    return ProfitStats(
        median=float(np.median(final_values)),
        mean=float(np.mean(final_values)),
        p5=float(np.percentile(final_values, 5)),
        p95=float(np.percentile(final_values, 95)),
        worst=float(np.min(final_values)),
        best=float(np.max(final_values)),
        prob_profit=float(np.mean(final_values > amount)),
    )


@dataclass
class ProfitResult:
    ticker: str
    months: int
    amount: float
    trials: int
    window_start: str
    window_end: str
    num_available_months: int
    sampling: str
    strat: ProfitStats
    bh: ProfitStats
    example: list[tuple[str, float, float]]  # (month, strat_ret, bh_ret) for one draw


def simulate_profit(
    strat_m: pd.Series,
    bh_m: pd.Series,
    months: int = 12,
    amount: float = 1000.0,
    trials: int = 1000,
    seed: int | None = None,
) -> tuple[ProfitStats, ProfitStats, str, list[tuple[str, float, float]]]:
    """Draw ``months`` random months ``trials`` times and compound ``amount``.

    Returns (strategy stats, buy&hold stats, sampling-mode label, example draw).
    """
    if len(strat_m) == 0:
        raise ValueError("no months available in the chosen window")
    rng = np.random.default_rng(seed)
    n = len(strat_m)
    replace = n < months
    sampling = "bootstrap (with replacement)" if replace else "without replacement"

    strat_arr = strat_m.to_numpy()
    bh_arr = bh_m.to_numpy()
    strat_final = np.empty(trials)
    bh_final = np.empty(trials)
    example: list[tuple[str, float, float]] = []

    for k in range(trials):
        idx = rng.choice(n, size=months, replace=replace)
        strat_final[k] = amount * float(np.prod(1.0 + strat_arr[idx]))
        bh_final[k] = amount * float(np.prod(1.0 + bh_arr[idx]))
        if k == 0:
            order = np.sort(idx)
            example = [
                (str(strat_m.index[i]), float(strat_arr[i]), float(bh_arr[i]))
                for i in order
            ]

    return _stats(strat_final, amount), _stats(bh_final, amount), sampling, example


def run_profit(
    ticker: str | None = None,
    prices: pd.Series | None = None,
    period: str | None = None,
    csv: str | None = None,
    months: int = 12,
    years: int = 5,
    start: str | None = None,
    end: str | None = None,
    amount: float = 1000.0,
    trials: int = 1000,
    seed: int | None = None,
    window: int = DEFAULT_WINDOW,
    bull_threshold: float = DEFAULT_BULL_THRESHOLD,
    bear_threshold: float = DEFAULT_BEAR_THRESHOLD,
    leverage: float = 1.0,
    max_exposure: float = 1.0,
    allow_short: bool = True,
    vol_target: float | None = None,
    max_leverage: float = 3.0,
    jump_penalty: float = 0.0,
) -> ProfitResult:
    if prices is None:
        # Pull extra history so the walk-forward is warmed up before the window.
        if period is None:
            period = f"{years + 2}y"
        prices = load_prices(ticker, period=period, csv=csv)

    strat_m, bh_m = monthly_returns(
        prices, window, bull_threshold, bear_threshold, leverage, max_exposure,
        allow_short, vol_target, max_leverage, jump_penalty,
    )

    # Resolve the [start, end] window as monthly periods.
    last = strat_m.index.max()
    end_p = pd.Period(end, "M") if end else last
    start_p = pd.Period(start, "M") if start else end_p - (12 * years - 1)
    in_window = strat_m[(strat_m.index >= start_p) & (strat_m.index <= end_p)]
    bh_window = bh_m.reindex(in_window.index)

    s_stats, b_stats, sampling, example = simulate_profit(
        in_window, bh_window, months=months, amount=amount, trials=trials, seed=seed
    )

    return ProfitResult(
        ticker=str(ticker or getattr(prices, "name", "series")),
        months=months,
        amount=amount,
        trials=trials,
        window_start=str(in_window.index.min()) if len(in_window) else str(start_p),
        window_end=str(in_window.index.max()) if len(in_window) else str(end_p),
        num_available_months=len(in_window),
        sampling=sampling,
        strat=s_stats,
        bh=b_stats,
        example=example,
    )


def format_profit(r: ProfitResult) -> str:
    amt = r.amount

    def _pct(v: float) -> str:
        return f"{(v / amt - 1.0) * 100:+.1f}%"

    out: list[str] = []
    out.append("=" * 72)
    out.append(f"  PROFIT MODE — {r.ticker}")
    out.append("=" * 72)
    out.append(
        f"invest ${amt:,.0f}  across {r.months} random months  "
        f"window {r.window_start}..{r.window_end}"
    )
    out.append(
        f"{r.num_available_months} months available  "
        f"{r.trials:,} trials  sampling: {r.sampling}"
    )
    out.append("")
    hdr = f"  {'outcome ($ / return)':<26}{'STRATEGY':>20}{'BUY & HOLD':>20}"
    out.append(hdr)
    out.append("  " + "-" * 66)

    def _row(label: str, sg: float, bg: float) -> str:
        return (
            f"  {label:<26}"
            f"{f'${sg:,.0f} ({_pct(sg)})':>20}"
            f"{f'${bg:,.0f} ({_pct(bg)})':>20}"
        )

    s, b = r.strat, r.bh
    out.append(_row("median", s.median, b.median))
    out.append(_row("mean", s.mean, b.mean))
    out.append(_row("5th percentile", s.p5, b.p5))
    out.append(_row("95th percentile", s.p95, b.p95))
    out.append(_row("worst trial", s.worst, b.worst))
    out.append(_row("best trial", s.best, b.best))
    out.append(
        f"  {'P(profit)':<26}{f'{s.prob_profit * 100:.0f}%':>20}"
        f"{f'{b.prob_profit * 100:.0f}%':>20}"
    )
    out.append("")
    out.append(f"Example draw of {r.months} months (first trial):")
    out.append(f"  {'month':<10}{'strategy':>12}{'buy & hold':>14}")
    for month, sr, br in r.example:
        out.append(f"  {month:<10}{sr * 100:>11.1f}%{br * 100:>13.1f}%")
    out.append("=" * 72)
    out.append("NOT financial advice — educational reproduction of a YouTube method.")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quant.profit",
        description="Monte-Carlo profit of the regime strategy over random months.",
    )
    p.add_argument("ticker", nargs="?", help="ticker symbol (live via yfinance)")
    p.add_argument("--csv", help="load prices from a CSV instead of yfinance (offline)")
    p.add_argument("--months", type=int, default=12, help="months to sample (default: 12)")
    p.add_argument("--years", type=int, default=5, help="window length in years (default: 5)")
    p.add_argument("--start", help="window start as YYYY-MM (overrides --years)")
    p.add_argument("--end", help="window end as YYYY-MM (default: latest)")
    p.add_argument("--amount", type=float, default=1000.0, help="money to invest (default: 1000)")
    p.add_argument("--trials", type=int, default=1000, help="Monte-Carlo trials (default: 1000)")
    p.add_argument("--seed", type=int, help="random seed for reproducibility")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--max-exposure", type=float, default=1.0)
    p.add_argument("--long-only", action="store_true", help="de-risk only; no net shorts")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ticker and not args.csv:
        build_parser().error("provide a ticker or --csv")
    try:
        result = run_profit(
            ticker=args.ticker,
            csv=args.csv,
            months=args.months,
            years=args.years,
            start=args.start,
            end=args.end,
            amount=args.amount,
            trials=args.trials,
            seed=args.seed,
            window=args.window,
            leverage=args.leverage,
            max_exposure=args.max_exposure,
            allow_short=not args.long_only,
        )
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_profit(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
