"""Price-data loading: live via yfinance, or offline from a CSV.

The engine only ever needs a single ``pd.Series`` of closing prices indexed by
date. ``load_prices`` is the one entry point the rest of the package uses; it
dispatches to yfinance for live tickers or to a local CSV (handy for tests and
for running where outbound network is blocked).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_yfinance(ticker: str, period: str = "10y", interval: str = "1d") -> pd.Series:
    """Download closing prices for ``ticker`` using yfinance.

    yfinance is an optional dependency (``pip install '.[quant]'``); we import it
    lazily so the rest of the engine works offline.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "yfinance is required for live data. "
            "Install it with: pip install 'ray-backend[quant]' "
            "(or: pip install yfinance)."
        ) from exc

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"yfinance returned no data for ticker {ticker!r}")

    close = df["Close"]
    # yfinance may return a single-column DataFrame for one ticker.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = ticker
    return close


def load_csv(path: str | Path, column: str = "Close") -> pd.Series:
    """Load closing prices from a CSV with a date index and a price column.

    Tries to auto-detect a date column among the first column / ``Date`` /
    ``date`` and a price column (``column``, else ``Adj Close``, else ``Close``).
    """
    df = pd.read_csv(path)
    # Identify the date column.
    date_col = next(
        (c for c in ("Date", "date", df.columns[0]) if c in df.columns),
        df.columns[0],
    )
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    price_col = next(
        (c for c in (column, "Adj Close", "Close", "close") if c in df.columns),
        None,
    )
    if price_col is None:
        raise ValueError(
            f"CSV {path} has no usable price column "
            f"(looked for {column!r}, 'Adj Close', 'Close')"
        )
    close = df[price_col].dropna()
    close.name = Path(path).stem
    return close


def load_prices(
    ticker: str | None = None,
    period: str = "10y",
    interval: str = "1d",
    csv: str | Path | None = None,
) -> pd.Series:
    """Load a close-price Series, preferring a local CSV when provided."""
    if csv is not None:
        return load_csv(csv)
    if ticker is None:
        raise ValueError("provide either a ticker (live) or a csv path (offline)")
    return load_yfinance(ticker, period, interval)
