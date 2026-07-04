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
    "&page=List_of_S%26P_500_companies&prop=wikitext&format=json&formatversion=2"
)

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
    lower = wikitext.lower()
    start = lower.find("selected changes")
    if start == -1:
        return []
    section = wikitext[start:]
    t_start = section.find("{|")
    t_end = section.find("\n|}", t_start)
    if t_start == -1 or t_end == -1:
        return []
    table = section[t_start:t_end]

    events: list[AdditionEvent] = []
    for chunk in table.split("\n|-")[1:]:
        line = chunk.strip()
        if line.startswith("!"):  # header row
            continue
        if line.startswith("|"):
            line = line[1:]
        cells = line.split("||")
        if len(cells) < 5:
            continue
        effective = _parse_date(_strip_markup(cells[0]))
        added = _strip_markup(cells[1])
        if effective is None or not _TICKER_RE.match(added):
            continue
        reason = _strip_markup(cells[5]) if len(cells) > 5 else ""
        m = re.search(
            r"\|\s*date\s*=\s*([A-Z][a-z]+ \d{1,2},? \d{4})", chunk
        )
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


def fetch_wikitext(cache_path: str | None = None) -> str:
    """Fetch the article wikitext via the MediaWiki API (cached to disk if given)."""
    import json
    import os
    import subprocess

    if cache_path and os.path.exists(cache_path):
        return json.load(open(cache_path))["parse"]["wikitext"]

    ca = "/root/.ccr/ca-bundle.crt"
    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    args = ["curl", "-sS", "-H", f"User-Agent: {ua}"]
    if os.path.exists(ca):
        args += ["--cacert", ca]
    args.append(WIKI_API)
    out = subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    if cache_path:
        with open(cache_path, "w") as fh:
            fh.write(out)
    return json.loads(out)["parse"]["wikitext"]


def load_events(cache_path: str | None = None) -> list[AdditionEvent]:
    """Convenience: fetch + parse the current S&P 500 change table."""
    return parse_changes(fetch_wikitext(cache_path))
