# Markov Regime "Hedge Fund Method" Quant Engine

A self-contained, educational reproduction of the 10-element **Markov regime
trading method** (the "hedge fund method" from the Lewis Jackson / "Rowan"
transcript). It quantifies a market's recent behaviour into three regimes, builds
a Markov transition matrix, forecasts forward, generates a position-sizing
signal, and validates everything with a walk-forward backtest and a Hidden
Markov Model cross-check.

> ⚠️ **NOT financial advice.** This is an educational reproduction of a method
> described in a YouTube video. It is not a production trading system and makes
> no guarantees about real-world performance.

## The 10 elements → code

| # | Element | Module |
|---|---------|--------|
| 1 | States (bull / sideways / bear) | `states.py` |
| 2 | Label every day's state | `states.py` |
| 3 | Markov property (only *today* drives tomorrow) | design — the signal reads only the current row of `P` |
| 4 | 3×3 transition matrix | `transition.py` |
| 5 | Persistence / stickiness (the diagonal) | `transition.py` |
| 6 | Squaring the matrix (n-day forecast) | `transition.py` — `n_step` |
| 7 | Stationary distribution (long-run mix) | `transition.py` — `stationary_distribution` |
| 8 | Signal = P(bull) − P(bear), sized by magnitude | `signals.py` |
| 9 | Walk-forward backtest (no look-ahead) | `backtest.py` |
| 10| Hidden Markov Model cross-check | `hmm.py` |

`regime.py` ties it together (`MarkovRegimeModel`), `cli.py` prints a report, and
`data.py` loads prices (live via yfinance or offline from a CSV).

## How a regime is defined

Sum the daily percent returns over the trailing **20** trading days:

- `>= +5%` → **Bull**
- `<= −5%` → **Bear**
- otherwise → **Sideways**

(Window and thresholds are configurable.) Every day from day 20 onward is
labeled, every consecutive `(today → tomorrow)` pair is tallied, and the tallies
are row-normalized into the transition matrix `P` where `P[i, j] =
P(tomorrow = j | today = i)` and each row sums to 1.

The trade signal is simply `P(bull tomorrow) − P(bear tomorrow)` for today's
state: the **sign** is the direction (long / short) and the **magnitude** is the
conviction → position size.

## Install

```bash
# from the repo root
pip install -e '.[quant]'   # adds yfinance + hmmlearn (numpy/scipy/pandas already core)
```

## Usage

### CLI

```bash
# Live data via yfinance
quant-markov SPY --period 10y
python -m quant BTC-USD --period 5y
python -m quant TSLA --window 20 --bull 0.05 --bear -0.05

# Offline from a CSV (Date + Close columns)
python -m quant --csv path/to/prices.csv

# Faster: skip the backtest and/or HMM
python -m quant SPY --no-backtest --no-hmm
```

The report shows the transition matrix, per-state stickiness, multi-horizon
forecasts (+1/+2/+3/+28 days via matrix powers), the stationary distribution,
today's state + signal + target position, the walk-forward backtest vs buy &
hold, and the HMM agreement rate.

### Library

```python
from quant import MarkovRegimeModel

model = MarkovRegimeModel("SPY", window=20, bull_threshold=0.05, bear_threshold=-0.05)
report = model.analyze(period="10y")

print(report.current_state_name, report.current_signal)
print(report.transition_matrix)
print(report.backtest.sharpe)
```

You can also pass your own price Series and bypass data loading:

```python
report = model.analyze(prices=my_close_series)
```

## Tests

The test suite uses deterministic synthetic price series and needs **no
network**:

```bash
pytest quant/tests -v
```

The HMM tests auto-skip if `hmmlearn` isn't installed.

## Bonus study: the S&P 500 "index effect" (`index_effect/`)

A self-contained empirical test of whether index-inclusion front-running still
pays a *small* trader (no market impact), or has been arbitraged away as the
"common wisdom" holds. It parses S&P 500 additions (with real **announcement**
vs **effective** dates) from Wikipedia, pulls daily prices from Nasdaq/Yahoo,
and runs a market-adjusted event study.

**Headline (2021–2026, 38 announcement-dated adds):** buying at the announcement
close and selling into the rebalance earns **~+6.5% market-adjusted (t≈6, 79%
win), positive every year** and survives the repo's own Deflated-Sharpe
multiple-testing guardrail (DSR≈1.0) — the effect is *not* dead. But ~4.5% of it
is an overnight gap at the (surprise — no pre-announcement drift) public
announcement, so it rewards *prediction* / prompt execution, not laziness. The
residuals a lazy small trader could get on public info alone — the next-open long
(~+1.9%) and shorting the post-rebalance reversal in **illiquid** additions
(~+3.5%/20d) — are nominally positive but *fail* deflation (DSR≈0.4). Full
write-up and caveats: [`index_effect/FINDINGS.md`](index_effect/FINDINGS.md).

```bash
# reproduce (needs network the first time; then cached)
PYTHONPATH=<repo parent> python -m quant.index_effect.report
pytest tests/test_index_effect.py -v   # engine is unit-tested offline
```

## Notes

- **Walk-forward** rebuilds the states + matrix from scratch on every day using
  only data up to that day, so no future information leaks into a decision
  (element 9). It's the honest way to backtest this.
- The **stationary distribution** is what `P^n` converges to as `n` grows — at a
  long horizon (e.g. 28 days) the forecast flattens toward it and carries no
  actionable signal (element 7).
- The **HMM** discovers regimes unsupervised, with no `±5%` threshold; where its
  labels agree with the rule-based labels you get the transcript's "green light"
  (element 10).
