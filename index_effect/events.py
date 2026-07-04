"""S&P 500 constituent-change events.

Parses the "Selected changes to the list of S&P 500 components" table from the
Wikipedia article (fetched as wikitext via the MediaWiki API) into a list of
:class:`AdditionEvent` records. Crucially, the citation ``<ref ... date=...>``
attached to each row usually carries the S&P Dow Jones Indices *announcement*
date, which is distinct from — and typically a handful of trading days before —
the *effective* date on which the change takes place. Both matter: the pop from
index inclusion is an announcement-driven event, and the forced index-fund
rebalance executes into the close preceding the effective date.

Network access lives in :func:`fetch_wikitext`; :func:`parse_changes` is a pure
function over wikitext and is unit-tested with a static fixture (no network).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

WIKI_API = (
    "https://en.wikipedia.org/w/api.php?action=parse"
    "&page={page}&prop=wikitext&format=json&formatversion=2"
)

# Wikipedia articles whose constituent-change tables share the S&P/Nasdaq
# ``id="changes"`` format (Date | Added ticker/security | Removed … | Reason),
# mapped to the ETF that tracks the index. These are the indices with a
# predictable, scheduled reconstitution *and* a publicly dated change log.
# (The Dow is deliberately excluded: it is committee-selected and changes only a
# handful of times a decade — not a scheduled, rules-based reconstitution, so it
# doesn't fit the "predictable rebalancing" premise.)
INDEX_PAGES = {
    "S&P 500": ("List of S&P 500 companies", "SPY"),
    "S&P 400 MidCap": ("List of S&P 400 companies", "MDY"),
    "S&P 600 SmallCap": ("List of S&P 600 companies", "IJR"),
    "Nasdaq-100": ("Nasdaq-100", "QQQ"),
}

_TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,5}$")
_DATE_FMTS = ("%B %d, %Y", "%B %d %Y", "%Y-%m-%d")


@dataclass(frozen=True)
class AdditionEvent:
    """A single stock addition to the S&P 500.

    ``announcement`` is ``None`` when the source row carried no datable citation.
    ``lag_calendar_days`` is announcement→effective in calendar days (``None`` if
    the announcement date is unknown).
    """

    ticker: str
    effective: dt.date
    announcement: dt.date | None = None
    reason: str = ""

    @property
    def lag_calendar_days(self) -> int | None:
        if self.announcement is None:
            return None
        return (self.effective - self.announcement).days


def _parse_date(text: str) -> dt.date | None:
    text = text.strip()
    for fmt in _DATE_FMTS:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _strip_markup(cell: str) -> str:
    """Collapse ``[[Foo|Bar]]``/``[[Bar]]`` wikilinks and drop self-closing refs."""
    cell = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", cell)
    cell = re.sub(r"<ref[^>]*/>", "", cell)
    return cell.strip()


def parse_changes(wikitext: str) -> list[AdditionEvent]:
    """Extract addition events from the article wikitext.

    Only rows with a non-empty *added* ticker become events (S&P always records a
    change as "AAA replaces BBB", so a row can add, remove, or both). The
    announcement date, when present, is read from the first ``date=`` field of
    the row's citation.
    """
    # Prefer the canonical id="changes" table (S&P 500/400/600, Nasdaq-100, Dow);
    # fall back to a "selected changes" heading for older/other layouts.
    pos = wikitext.find('id="changes"')
    if pos != -1:
        t_start = wikitext.rfind("{|", 0, pos)
    else:
        start = wikitext.lower().find("selected changes")
        if start == -1:
            return []
        t_start = wikitext.find("{|", start)
    if t_start == -1:
        return []
    t_end = wikitext.find("\n|}", t_start)
    if t_end == -1:
        return []
    table = wikitext[t_start:t_end]

    events: list[AdditionEvent] = []
    for chunk in table.split("\n|-")[1:]:
        # Cells may be inline (``a || b || c``) or one-per-line (``\n|a\n|b``);
        # handle both so the parser works across S&P and Nasdaq/Dow layouts.
        cells: list[str] = []
        for ln in chunk.split("\n"):
            s = ln.strip()
            if not s or s.startswith("!"):  # blank or header cell
                continue
            if s.startswith("|"):
                cells.extend(s[1:].split("||"))
        if len(cells) < 5:
            continue
        effective = _parse_date(_strip_markup(cells[0]))
        added = _strip_markup(cells[1])
        if effective is None or not _TICKER_RE.match(added):
            continue
        reason = _strip_markup(cells[5]) if len(cells) > 5 else ""
        m = re.search(
            r"\|\s*date\s*=\s*([A-Z][a-z]+ \d{1,2},? \d{4})", chunk
        ) or re.search(r"\|\s*date\s*=\s*(\d{4}-\d{2}-\d{2})", chunk)
        announcement = _parse_date(m.group(1)) if m else None
        # Guard against citations that reference the *effective* announcement of a
        # later, unrelated change: only trust an announcement strictly before ED
        # and within a sane window.
        if announcement is not None and not (0 < (effective - announcement).days <= 45):
            announcement = None
        events.append(
            AdditionEvent(
                ticker=added,
                effective=effective,
                announcement=announcement,
                reason=reason.split("<ref")[0].strip(),
            )
        )
    events.sort(key=lambda e: e.effective)
    return events


def fetch_wikitext(
    page: str = "List of S&P 500 companies", cache_path: str | None = None
) -> str:
    """Fetch an article's wikitext via the MediaWiki API (cached to disk if given)."""
    import json
    import os
    import subprocess
    import urllib.parse

    if cache_path and os.path.exists(cache_path):
        return json.load(open(cache_path))["parse"]["wikitext"]

    ca = "/root/.ccr/ca-bundle.crt"
    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    args = ["curl", "-sS", "-H", f"User-Agent: {ua}"]
    if os.path.exists(ca):
        args += ["--cacert", ca]
    args.append(WIKI_API.format(page=urllib.parse.quote(page)))
    out = subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    if cache_path:
        with open(cache_path, "w") as fh:
            fh.write(out)
    return json.loads(out)["parse"]["wikitext"]


def load_events(
    page: str = "List of S&P 500 companies", cache_path: str | None = None
) -> list[AdditionEvent]:
    """Fetch + parse an index's constituent-change table into addition events."""
    return parse_changes(fetch_wikitext(page, cache_path))
