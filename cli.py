"""Command-line entry point: ``python -m quant SPY --period 10y``.

Prints a human-readable regime report — the 3x3 transition matrix, stickiness,
multi-horizon forecasts, the stationary (long-run) mix, today's state and
signal, the walk-forward backtest summary vs buy & hold, and the HMM agreement.
"""

from __future__ import annotations

import argparse
import sys

from .regime import DEFAULT_HORIZONS, MarkovRegimeModel, RegimeReport
from .states import STATE_NAMES


def _fmt_matrix(P, title: str) -> str:
    header = "            " + "".join(f"{n:>10}" for n in STATE_NAMES)
    lines = [title, header]
    for i, name in enumerate(STATE_NAMES):
        row = "".join(f"{P[i, j] * 100:9.1f}%" for j in range(P.shape[1]))
        lines.append(f"  {name:>9} {row}")
    return "\n".join(lines)


def _fmt_vector(vec, title: str) -> str:
    cells = "  ".join(f"{name}: {vec[i] * 100:5.1f}%" for i, name in enumerate(STATE_NAMES))
    return f"{title}\n  {cells}"


def format_report(report: RegimeReport) -> str:
    out: list[str] = []
    out.append("=" * 64)
    out.append(f"  MARKOV REGIME REPORT — {report.ticker}")
    out.append("=" * 64)
    out.append(
        f"window={report.window}d  "
        f"bull>=+{report.bull_threshold * 100:.0f}%  "
        f"bear<={report.bear_threshold * 100:.0f}%  "
        f"labeled days={report.num_labeled_days}"
    )
    out.append("")
    out.append(_fmt_matrix(report.transition_matrix, "Transition matrix  P(tomorrow | today):"))
    out.append("")
    sticky = "  ".join(
        f"{name}: {report.stickiness[i] * 100:.1f}%" for i, name in enumerate(STATE_NAMES)
    )
    out.append(f"Stickiness (persistence / diagonal):\n  {sticky}")
    out.append("")
    out.append(_fmt_vector(report.stationary_distribution, "Stationary (long-run) distribution:"))
    out.append("")
    out.append("Multi-horizon forecasts P(state in n days | today):")
    cur = report.current_state
    for n, M in report.horizon_forecasts.items():
        row = M[cur]
        cells = "  ".join(f"{name}: {row[j] * 100:5.1f}%" for j, name in enumerate(STATE_NAMES))
        out.append(f"  +{n:>2}d  {cells}")
    out.append("")
    out.append("-" * 64)
    direction = "LONG" if report.current_signal > 0 else "SHORT" if report.current_signal < 0 else "FLAT"
    out.append(f"Today's state : {report.current_state_name}")
    out.append(
        f"Signal        : {report.current_signal * 100:+.1f}%  "
        f"(P(bull) - P(bear))  ->  {direction}"
    )
    out.append(f"Target position: {report.current_position * 100:+.1f}% exposure")
    out.append("-" * 64)

    if report.backtest is not None:
        bt = report.backtest
        out.append("")
        out.append("Walk-forward backtest (no look-ahead):")
        out.append(f"  traded days        : {bt.num_days}")
        out.append(f"  total return       : {bt.total_return * 100:+.1f}%")
        out.append(f"  annualized return  : {bt.annualized_return * 100:+.1f}%")
        out.append(f"  annualized vol     : {bt.annualized_volatility * 100:.1f}%")
        out.append(f"  Sharpe             : {bt.sharpe:.2f}")
        out.append(f"  max drawdown       : {bt.max_drawdown * 100:.1f}%")
        out.append(f"  turnover (sum|dpos|): {bt.turnover:.1f}")
        out.append(f"  buy & hold return  : {bt.buy_hold_return * 100:+.1f}%")
    elif "backtest_error" in report.extras:
        out.append(f"\nBacktest skipped: {report.extras['backtest_error']}")

    if report.hmm is not None:
        out.append("")
        out.append("HMM cross-check (element 10):")
        out.append(f"  agreement with rule labels : {report.hmm.agreement_rate * 100:.1f}%")
        latest = report.hmm.confirmation.iloc[-1] if len(report.hmm.confirmation) else False
        out.append(f"  today confirmed by HMM     : {'YES (green light)' if latest else 'no'}")
    elif "hmm_error" in report.extras:
        out.append(f"\nHMM cross-check skipped: {report.extras['hmm_error']}")

    out.append("=" * 64)
    out.append("NOT financial advice — educational reproduction of a YouTube method.")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quant",
        description="Markov regime 'hedge fund method' analysis for any ticker.",
    )
    p.add_argument("ticker", nargs="?", help="ticker symbol, e.g. SPY, BTC-USD (live via yfinance)")
    p.add_argument("--csv", help="load prices from a CSV instead of yfinance (offline)")
    p.add_argument("--period", default="10y", help="yfinance period (default: 10y)")
    p.add_argument("--window", type=int, default=20, help="look-back window in days (default: 20)")
    p.add_argument("--bull", type=float, default=0.05, help="bull threshold (default: 0.05)")
    p.add_argument("--bear", type=float, default=-0.05, help="bear threshold (default: -0.05)")
    p.add_argument("--leverage", type=float, default=1.0, help="signal leverage (default: 1.0)")
    p.add_argument("--max-exposure", type=float, default=1.0, help="position clip (default: 1.0)")
    p.add_argument(
        "--vol-target",
        type=float,
        help="annualized vol target, e.g. 0.15 (scales position to constant vol)",
    )
    p.add_argument("--max-leverage", type=float, default=3.0, help="leverage cap with --vol-target")
    p.add_argument(
        "--hysteresis",
        type=float,
        default=0.0,
        help="no-trade buffer: hold prior position unless target moves by more than this",
    )
    p.add_argument(
        "--jump-penalty",
        type=float,
        default=0.0,
        help="jump-model switch penalty on regime labels (0 = greedy threshold labeling)",
    )
    p.add_argument(
        "--long-only",
        action="store_true",
        help="floor positions at 0 (de-risk only; no net shorts in bear regimes)",
    )
    p.add_argument(
        "--profit",
        action="store_true",
        help="profit mode: Monte-Carlo $ outcome over random months (see --months/--amount)",
    )
    p.add_argument("--months", type=int, default=12, help="profit mode: months to sample (12)")
    p.add_argument("--years", type=int, default=5, help="profit mode: window length in years (5)")
    p.add_argument("--start", help="profit mode: window start YYYY-MM (overrides --years)")
    p.add_argument("--end", help="profit mode: window end YYYY-MM (default: latest)")
    p.add_argument("--amount", type=float, default=1000.0, help="profit mode: $ to invest (1000)")
    p.add_argument("--trials", type=int, default=1000, help="profit mode: Monte-Carlo trials (1000)")
    p.add_argument("--seed", type=int, help="profit mode: random seed")
    p.add_argument("--no-backtest", action="store_true", help="skip the walk-forward backtest")
    p.add_argument("--no-hmm", action="store_true", help="skip the HMM cross-check")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ticker and not args.csv:
        build_parser().error("provide a ticker or --csv")

    if args.profit:
        from .profit import format_profit, run_profit  # noqa: PLC0415

        try:
            result = run_profit(
                ticker=args.ticker,
                csv=args.csv,
                period=args.period if args.csv is None else None,
                months=args.months,
                years=args.years,
                start=args.start,
                end=args.end,
                amount=args.amount,
                trials=args.trials,
                seed=args.seed,
                window=args.window,
                bull_threshold=args.bull,
                bear_threshold=args.bear,
                leverage=args.leverage,
                max_exposure=args.max_exposure,
                allow_short=not args.long_only,
            )
        except (ValueError, ImportError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_profit(result))
        return 0

    model = MarkovRegimeModel(
        ticker=args.ticker,
        window=args.window,
        bull_threshold=args.bull,
        bear_threshold=args.bear,
    )
    try:
        report = model.analyze(
            period=args.period,
            csv=args.csv,
            horizons=DEFAULT_HORIZONS,
            run_backtest=not args.no_backtest,
            run_hmm=not args.no_hmm,
            leverage=args.leverage,
            max_exposure=args.max_exposure,
            allow_short=not args.long_only,
            vol_target=args.vol_target,
            max_leverage=args.max_leverage,
            hysteresis=args.hysteresis,
            jump_penalty=args.jump_penalty,
        )
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
