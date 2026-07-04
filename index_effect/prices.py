"""Daily price loaders for the index-effect study.

Two free sources, both reached through the environment's HTTPS proxy:

* **Nasdaq** (``api.nasdaq.com``) — the primary source. Fast, unthrottled, and
  returns OHLCV, but only ~5 years of history. That window (roughly 2021 →
  present) is exactly the "modern" regime in which the common wisdom claims the
  index effect has been arbitraged away, so it is the sample of interest.
* **Yahoo** (``query{1,2}.finance.yahoo.com``) — a fallback with adjusted closes
  and deep history, but aggressively rate-limits datacenter IPs.

Both cache raw JSON to disk so a study is reproducible offline after one fetch.
Returned frames are ``pandas`` with a ``DatetimeIndex`` and columns
``close, open, volume`` (Nasdaq close is unadjusted; over a ±one-month event
window dividend drift is negligible and splits are guarded in the study).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time

import pandas as pd

CA = "/root/.ccr/ca-bundle.crt"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "_pricecache")


def _curl(url: str, extra_headers: list[str] | None = None, timeout: int = 60) -> str:
    args = ["curl", "-sS", "-H", f"User-Agent: {UA}"]
    if os.path.exists(CA):
        args += ["--cacert", CA]
    for h in extra_headers or []:
        args += ["-H", h]
    args.append(url)
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout


def _num(x) -> float:
    return float(str(x).replace("$", "").replace(",", "").strip())


def _frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "close", "open", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_nasdaq(
    ticker: str,
    start: str = "2021-06-01",
    end: str = "2026-07-04",
    assetclass: str = "stocks",
    cache_dir: str = DEFAULT_CACHE,
    tries: int = 4,
    fallback: bool = True,
) -> pd.DataFrame | None:
    """Load daily OHLCV from Nasdaq. Returns ``None`` if unavailable/delisted.

    ``fallback=False`` skips the stocks→etf retry (faster when scanning many
    tickers of which some are simply delisted).
    """
    os.makedirs(cache_dir, exist_ok=True)
    fp = os.path.join(cache_dir, f"nasdaq_{ticker}_{assetclass}.json".replace("/", "_"))
    if os.path.exists(fp):
        cached = json.load(open(fp))
        return None if isinstance(cached, dict) else _frame(cached)

    url = (
        f"https://api.nasdaq.com/api/quote/{ticker}/historical"
        f"?assetclass={assetclass}&fromdate={start}&todate={end}&limit=9999"
    )
    for i in range(tries):
        try:
            data = json.loads(_curl(url, ["Accept: application/json"])).get("data") or {}
        except Exception:
            time.sleep(2 * (i + 1))
            continue
        table = data.get("tradesTable") or {}
        rows = table.get("rows")
        if not rows:
            if fallback and assetclass == "stocks" and i == 0:
                return load_nasdaq(ticker, start, end, "etf", cache_dir, tries=2)
            time.sleep(0.5 * (i + 1))
            continue
        parsed = []
        for r in rows:
            try:
                parsed.append(
                    [
                        dt.datetime.strptime(r["date"], "%m/%d/%Y").date().isoformat(),
                        _num(r["close"]),
                        _num(r.get("open") or r["close"]),
                        _num(r.get("volume") or 0),
                    ]
                )
            except Exception:
                continue
        if parsed:
            parsed.sort()
            json.dump(parsed, open(fp, "w"))
            return _frame(parsed)
        time.sleep(1.5 * (i + 1))
    json.dump({"__error__": "unavailable"}, open(fp, "w"))
    return None


_YAHOO_CRUMB: list = [None]  # lazily established (cookie jar + crumb)


def _yahoo_session() -> str | None:
    """Return a Yahoo crumb, establishing cookies first. The anonymous chart
    endpoint gets 429-throttled hard; an authenticated (cookie+crumb) session is
    not. Cookies are written to a jar the subsequent _curl calls reuse."""
    if _YAHOO_CRUMB[0] is not None:
        return _YAHOO_CRUMB[0] or None
    jar = os.path.join(DEFAULT_CACHE, "_yahoo_cookies.txt")
    os.makedirs(DEFAULT_CACHE, exist_ok=True)
    base = ["curl", "-sS", "-H", f"User-Agent: {UA}", "-c", jar, "-b", jar]
    if os.path.exists(CA):
        base += ["--cacert", CA]
    def _valid(c: str) -> bool:
        return bool(c) and " " not in c and "<" not in c and 6 <= len(c) <= 24

    for attempt in range(5):
        for u in ("https://finance.yahoo.com/quote/SPY", "https://fc.yahoo.com"):
            subprocess.run(base + [u], capture_output=True, text=True, timeout=30)
        crumb = subprocess.run(
            base + ["https://query1.finance.yahoo.com/v1/test/getcrumb"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        if _valid(crumb):
            _YAHOO_CRUMB[0] = crumb
            return crumb
        time.sleep(5 * (attempt + 1))  # getcrumb itself is rate-limited when hammered
    _YAHOO_CRUMB[0] = ""  # give up; caller falls back to anonymous (likely 429)
    return None


def load_yahoo(
    ticker: str,
    start: dt.date,
    end: dt.date,
    cache_dir: str = DEFAULT_CACHE,
    tries: int = 5,
) -> pd.DataFrame | None:
    """Load daily OHLCV from Yahoo's chart API via an authenticated session.

    ``close`` is the **raw** close (matching Nasdaq's semantics, for the pocket
    P&L); an ``adjclose`` column carries the dividend/split-adjusted series.
    Returns ``None`` if unavailable.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fp = os.path.join(cache_dir, f"yahoo_{ticker}_{start}_{end}.json".replace("/", "_"))
    if os.path.exists(fp):
        cached = json.load(open(fp))
        return None if isinstance(cached, dict) else _frame_adj(cached)

    crumb = _yahoo_session()
    jar = os.path.join(DEFAULT_CACHE, "_yahoo_cookies.txt")

    def unix(d):
        return int(time.mktime(dt.date(d.year, d.month, d.day).timetuple()))

    p1, p2 = unix(start), unix(end) + 86400
    for i in range(tries):
        host = "query1" if i % 2 == 0 else "query2"
        url = (
            f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={p1}&period2={p2}&interval=1d&events=div"
            + (f"&crumb={crumb}" if crumb else "")
        )
        args = ["curl", "-sS", "-H", f"User-Agent: {UA}", "-b", jar, "-c", jar]
        if os.path.exists(CA):
            args += ["--cacert", CA]
        args.append(url)
        try:
            j = json.loads(subprocess.run(args, capture_output=True, text=True, timeout=40).stdout)
        except Exception:
            time.sleep(3 * (i + 1))
            continue
        err = j.get("chart", {}).get("error")
        if err:
            if str(err.get("code")) in ("Not Found", "No data found, symbol may be delisted"):
                json.dump({"__error__": str(err)}, open(fp, "w"))
                return None
            time.sleep(4 * (i + 1))
            continue
        res = j["chart"]["result"][0]
        ts = res.get("timestamp", [])
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q.get("close"))
        rows = []
        for k, t in enumerate(ts):
            rawc = q["close"][k]
            if rawc is None:
                continue
            rows.append(
                [
                    dt.datetime.utcfromtimestamp(t).date().isoformat(),
                    float(rawc),  # raw close (pocket P&L)
                    float(q["open"][k]) if q.get("open") and q["open"][k] else None,
                    float(q["volume"][k]) if q.get("volume") and q["volume"][k] else None,
                    float(adj[k]) if adj and adj[k] is not None else float(rawc),  # adjclose
                ]
            )
        rows.sort()
        json.dump(rows, open(fp, "w"))
        return _frame_adj(rows)
    return None


def _frame_adj(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "close", "open", "volume", "adjclose"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()
