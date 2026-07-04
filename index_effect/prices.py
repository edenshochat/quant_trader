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
) -> pd.DataFrame | None:
    """Load daily OHLCV from Nasdaq. Returns ``None`` if unavailable/delisted."""
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
            if assetclass == "stocks" and i == 0:
                return load_nasdaq(ticker, start, end, "etf", cache_dir, tries=2)
            time.sleep(1.5 * (i + 1))
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


def load_yahoo(
    ticker: str,
    start: dt.date,
    end: dt.date,
    cache_dir: str = DEFAULT_CACHE,
    tries: int = 6,
    min_interval: float = 3.0,
    _last: list = [0.0],
) -> pd.DataFrame | None:
    """Load daily OHLCV from Yahoo's chart API (adjusted close). Paced + retried
    to survive rate limiting; returns ``None`` if unavailable."""
    os.makedirs(cache_dir, exist_ok=True)
    fp = os.path.join(cache_dir, f"yahoo_{ticker}_{start}_{end}.json".replace("/", "_"))
    if os.path.exists(fp):
        cached = json.load(open(fp))
        if not (isinstance(cached, dict) and "Too Many Requests" in str(cached)):
            return None if isinstance(cached, dict) else _frame(cached)

    def unix(d):
        return int(time.mktime(dt.date(d.year, d.month, d.day).timetuple()))

    p1, p2 = unix(start), unix(end) + 86400
    for i in range(tries):
        host = "query1" if i % 2 == 0 else "query2"
        url = (
            f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={p1}&period2={p2}&interval=1d&events=div"
        )
        gap = time.time() - _last[0]
        if gap < min_interval:
            time.sleep(min_interval - gap)
        _last[0] = time.time()
        try:
            j = json.loads(_curl(url))
        except Exception as exc:
            time.sleep((10 if "Too Many Requests" in str(exc) else 3) * (i + 1))
            continue
        err = j.get("chart", {}).get("error")
        if err:
            if str(err.get("code")) in ("Not Found", "No data found, symbol may be delisted"):
                json.dump({"__error__": str(err)}, open(fp, "w"))
                return None
            time.sleep(6 * (i + 1))
            continue
        res = j["chart"]["result"][0]
        ts = res.get("timestamp", [])
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q.get("close"))
        rows = []
        for k, t in enumerate(ts):
            c = adj[k] if adj and adj[k] is not None else q["close"][k]
            if c is None:
                continue
            rows.append(
                [
                    dt.datetime.utcfromtimestamp(t).date().isoformat(),
                    float(c),
                    float(q["open"][k]) if q.get("open") and q["open"][k] else None,
                    float(q["volume"][k]) if q.get("volume") and q["volume"][k] else None,
                ]
            )
        rows.sort()
        json.dump(rows, open(fp, "w"))
        return _frame(rows)
    return None
