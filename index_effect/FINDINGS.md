# Does the S&P 500 "index effect" still pay a small trader?

**Question.** A WhatsApp debate: can you profit by owning a stock *before* index
funds are forced to buy it at reconstitution, then selling into their
price-insensitive demand? "Common wisdom" (and ChatGPT, in the chat) says the
edge has been arbitraged away. This is an empirical test of that claim — at
*small* size, where market impact is negligible and the raw abnormal return over
a tradable window **is** the achievable edge.

> ⚠️ **NOT financial advice.** Educational research on historical data. Past
> abnormal returns are not a promise of future ones, and this ignores execution
> frictions beyond a first-order treatment. See caveats.

## Data & method

* **Events:** every S&P 500 *addition* parsed from Wikipedia's "Selected changes"
  table (`events.py`). Each row's citation usually carries the S&P Dow Jones
  Indices **announcement date (AD)**, which is distinct from the **effective
  date (ED)** — typically 2–11 trading days earlier and *variable* (so a fixed
  "ED−5" proxy would be wrong).
* **Sample:** the modern window our price source (Nasdaq, ~5y history) covers —
  **61 additions from Jul-2021 to Apr-2026, 38 with a datable announcement.**
  This is precisely the regime in which the effect is claimed dead.
* **Prices:** Nasdaq daily OHLCV; benchmark **SPY**. Returns are
  **market-adjusted** (stock minus SPY over the same window) — the standard
  robust choice for short event windows (per-name beta adds more noise than it
  removes). Splits inside a window are guarded.
* **Timing convention:** additions are effective at the **open** of ED; index
  funds execute the rebalance into the **close of ED−1**. S&P announces **after
  the close** on AD, so the market's first reaction is the AD→AD+1 session.

Reproduce: `PYTHONPATH=<repo parent> python -m quant.index_effect.report`
(engine is unit-tested offline in `tests/test_index_effect.py`).

## Headline result — the effect is alive

| Trade (market-adjusted)                     |  n |   mean |  median | t-stat | win% |
|---------------------------------------------|---:|-------:|--------:|-------:|-----:|
| Announcement day `C[AD-1]→C[AD]`            | 38 | +0.61% |  +0.77% |  +1.49 |  66% |
| **Buy `C[AD]` → sell into rebalance `C[ED-1]`** | 38 | **+6.52%** | **+6.27%** | **+5.89** | **79%** |
| Buy `C[AD]` → `C[ED]`                        | 38 | +6.24% |  +5.52% |  +5.75 |  82% |
| Buy next **`OPEN[AD+1]`** → `C[ED-1]`       | 38 | +1.89% |  +1.99% |  +2.41 |  63% |

The announcement→rebalance abnormal return averages **~6.5%** with a **t of ~6**
and a **79% hit rate**. Median ≈ mean and it survives dropping the three biggest
winners (**+5.24%, t=5.9**) and trimming both tails (**+5.99%, t=6.9**) — it is
**not** an outlier (COIN/SMCI/PLTR) artifact. It is **positive every calendar
year 2021–2026** (per-year t = 1.6 → 3.5). The "arbitraged to zero" claim is
**false** for this sample.

## But *when* does the money appear? (this settles the chat)

Decomposing the ~6.5%:

| Sub-window                          |   mean | t-stat | win% |
|-------------------------------------|-------:|-------:|-----:|
| Announcement day (during AD session)| +0.61% |  +1.5  | 66% |
| **Overnight gap `C[AD]→O[AD+1]`**   | **+4.5%** | **+6.9** | **87%** |
| Intraday AD+1 and drift → `C[ED-1]` | ~+1.4% |  ~+2   |  ~60% |

The bulk of the prize — **~4.5%** — is an **overnight gap** the morning after the
after-close announcement. So:

* **"I don't need visibility" is wrong.** ~70% of the edge reprices *instantly*
  when the announcement goes public. To capture the full 6.5% you must already
  be positioned *before* the after-close announcement — i.e. you need to
  **predict the addition** (the "visibility"/committee-forecast game the chat
  waved away), or transact in the after-hours print the moment it drops.
* **"It's fully priced in" is also wrong.** Acting *only* on public info and
  buying at the **next open**, a small trader still nets **+1.9% into the
  rebalance (t=2.4, 63% win)** — the forced-demand run-up is not entirely gone by
  the open. A small, real, no-foreknowledge edge remains.

The naive "the ETF buys from me at a ridiculous price" mental model is wrong
about *mechanism* (the pop is at announcement, not a lazy sell into the
rebalance) but right that **a mechanical, exploitable edge exists.**

## Is the addition anticipated? (pre-announcement drift)

Abnormal return in the 10 trading days *before* the announcement:

| Window                         |  n |   mean | median | t-stat | win% |
|--------------------------------|---:|-------:|-------:|-------:|-----:|
| `C[AD-10] → C[AD-1]` (pre-news)| 38 | +0.98% | +0.29% |  +0.79 |  53% |

**Not significant.** The addition is genuinely a *surprise* — there is no
tradable pre-announcement momentum/leakage drift; the ~4.5% jump is repricing
*at* the announcement, not gradual anticipation. Implication: the prediction
game must be **fundamental** (forecasting which eligible name the committee
picks), not a matter of riding price momentum into the event.

## The low-volume angle — a cleaner *short* edge

The user's instinct ("works at lower volume, no big money movements") shows up on
the **reversal** side. After the rebalance, added names give back gains — and the
give-back is concentrated in the **least-liquid** additions:

| Pre-event liquidity tercile | reversal `C[ED-1]→C[ED+20]` | SHORT P&L | t |
|-----------------------------|----------------------------:|----------:|--:|
| LOW  $vol ($42M–154M/day)   | −3.51% | **+3.51%** | −1.7 |
| MID  $vol ($165M–385M/day)  | −2.25% | +2.25% | −1.1 |
| HIGH $vol ($403M–11.5B/day) | +0.67% | −0.67% | +0.2 |

Shorting the freshly-added *illiquid* name after the forced buying earns ~+3.5%
over 20 days (marginally significant, t≈1.7); in liquid/momentum names (COIN,
PLTR, SMCI…) momentum dominates and there is no reversal. This is a
**capacity-limited, small-size** edge — big arbs can't work it without moving the
stock, which is exactly why a residual survives.

## How much survives multiple-testing correction?

Several windows were examined, so a lone t-stat flatters. Running each strategy's
per-trade returns through the repo's own guardrails — Probabilistic Sharpe (PSR,
P[true Sharpe>0]) and **Deflated Sharpe (DSR**, PSR against the expected best-of-N
noise Sharpe) from `quant.metrics`:

| Strategy (per-trade)              |  n | Sharpe | PSR   | **DSR** | survives? |
|-----------------------------------|---:|-------:|------:|--------:|:---------:|
| LONG prompt `C[AD]→ED-1`          | 38 | +0.97  | 1.000 | **≈1.00** | ✅ yes |
| LONG public-info `O[AD+1]→ED-1`   | 38 | +0.40  | 0.992 | **≈0.40** | ❌ no |
| SHORT illiquid reversal `ED-1→+20`| 20 | +0.39  | 0.974 | **≈0.40** | ❌ no |

(DSR uses ~10 examined configs; the exact figure depends on that assumption, but
the ranking is stable.) Only the **announcement-timed long** clears the bar
decisively. The two edges a *lazy* small trader could get on public info alone
are nominally significant (PSR>0.97) but **do not survive** deflation — they are
suggestive, not established.

## Verdict

* **Common wisdom (effect is dead): rejected.** Announcement→rebalance abnormal
  return ≈ **+6.5%, t≈6, DSR≈1.0**, every year 2021–2026 — a robust edge.
* **But it rewards *prediction* / prompt execution, not laziness.** ~4.5% is an
  overnight gap at the (surprise) public announcement, so you must already be
  positioned; there's no pre-announcement drift to ride in.
* **The residuals available on public info alone are marginal:** the next-open
  long (~+1.9%) and the illiquid-reversal short (~+3.5%/20d) are nominally
  positive but fail multiple-testing deflation (DSR≈0.4). Real-looking, not
  bankable.

So the chat's scheme isn't "stupid" — but the free-lunch framing is. The alpha is
real, it's front-loaded into the announcement, and the part that survives *at
small size with no foreknowledge* is a couple of percent long into the rebalance
plus a low-capacity reversal short.

## A real $100k pocket — concrete P&L and max drawdown

Turning the edge into an account. Fully executable, no leverage, no slippage
(small size): **BUY at `OPEN[AD+1]`** (market-on-open the session after the
after-close announcement — no foreknowledge), **SELL at `CLOSE[ED-1]`**
(market-on-close into the reconstitution). The pocket compounds on **raw**
returns; max drawdown is measured on the **daily mark-to-market** equity curve
over SPY's full calendar (flat-cash gaps and mid-hold dips included).

**All-in, one name at a time (Jan-2020 → Apr-2026, 50 trades):**

| metric | value |
|---|---|
| final equity (from $100k) | **$243k** (+143%, CAGR +15.4%) |
| win rate / avg trade | 62% / +1.94% |
| **worst single trade ("mistake")** | **APO −9.7%** |
| worst losing streak | −10.8% (CSGP → PCG) |
| **portfolio max drawdown** | **−17.8%** ($231k → $190k, Nov-2024 → Mar-2025) |
| longest underwater | **276 trading days (>1 yr)** |
| capital actually deployed | **~14% of the time** (in cash between events) |

Two things matter for "max drawdown when mistakes happen":

1. **It's a *cluster*, not one trade.** The −17.8% peak-to-trough was a run of
   weak trades in late-2024/early-2025 (APO −9.7% plus mark-to-market dips),
   and it took **over a year** to recover. Any single mistake caps near −10%.
2. **It's a dial you control.** Because opportunities are sparse and serial
   (you're invested only ~14% of the time), drawdown scales almost linearly with
   how much of the pocket you commit per trade:

| fraction per trade | final equity | max drawdown |
|--------------------:|-------------:|-------------:|
| 100% (all-in)       | $243k (+143%) | **−17.8%** |
| 50%                 | $159k (+59%)  | −9.3% |
| 25%                 | $127k (+27%)  | −4.7% |

Because holds are short (~1–2 weeks) and infrequent, the pocket carries little
market beta — it rode through the 2022 bear largely unscathed; the worst
drawdown was idiosyncratic (bad picks in late-2024/25), not the market. Adding
the 2020 → mid-2021 additions (TSLA, ETSY, ENPH …) roughly *doubled* the total
return versus the 2021-on subset but left the −17.8% max drawdown **unchanged**,
confirming the drawdown is a late-2024/25 event, not a data-window artifact.

> **Data note:** the 2020 → mid-2021 leg predates the Nasdaq API's ~5-year
> history, so those 14 names are pulled from Yahoo via an authenticated
> (cookie+crumb) session — the anonymous endpoint 429-throttles. Prices are raw
> (dividends negligible over the ~2-week holds; no in-window splits). Entry is
> the realistic no-foreknowledge open — the leg that *fails* Deflated-Sharpe
> above — so treat the +143% as illustrative of mechanics and the **drawdown
> profile (~−18% all-in, linear in size, >1yr recovery) as the durable
> takeaway.**

## Caveats

* Modern sample only (Nasdaq ≈5y); the pre-2021 decay comparison needs the Yahoo
  loader (`prices.load_yahoo`) once its rate limit clears.
* n = 38 announcement-dated events — solid t-stats, but a single-regime, single-
  source sample. Announcement dates are Wikipedia-sourced (internally consistent:
  the AD→AD+1 gap confirms after-close timing).
* Unadjusted closes (dividends negligible over ±1 month; splits guarded).
* Ignores borrow cost/availability on the short leg and after-hours slippage on
  the fast long entry; both are small for these names but non-zero.
* Multiple windows were examined; the headline is the *pre-registered* classic
  index-effect window, significant at t≈6 with per-year consistency — not the
  best of many p-hacked cuts.
