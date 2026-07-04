"""European index-inclusion events: DAX and TecDAX (Deutsche Börse / STOXX).

Unlike the S&P family and Nasdaq-100, no major European index (FTSE 100, DAX,
CAC 40, IBEX 35, FTSE MIB, AEX, SMI, EURO STOXX 50) has a Wikipedia article with
a structured, dated constituent-change table — only narrative "History"
prose, checked and confirmed empirically before writing this module. The Dow
was excluded from the US cross-index study for the same reason (committee-
selected, no scheduled reconstitution feed); the European indices above are
scheduled/rules-based but simply lack a *public, structured* changes log
reachable from this environment (STOXX itself gates it behind a JS-rendered,
anti-bot-protected site).

**STOXX does publish one**: an official "Historical Index Compositions" PDF
(https://www.stoxx.com/document/Indices/Common/Indexguide/Historical_Index_Compositions.pdf)
covering DAX, TecDAX, MDAX, and SDAX back to their 1987/2003 inception, in the
same (date of change, date of announcement, deletion, addition) shape as the
Wikipedia tables. This module parses that PDF's DAX and TecDAX sections (MDAX
and SDAX are scoped out — their far larger, more obscure small/mid-cap universe
makes reliable company-name -> ticker mapping impractical by hand).

Even within DAX/TecDAX, a handful of table rows couldn't be safely resolved:
multi-company blocks where the PDF's column layout is lost in text extraction
(no way to tell which name was added vs. removed) are dropped rather than
guessed — see ``extract_additions``'s ``dropped`` return value. Three
companies (ISRA VISION, Varta, SUSE S.A.) were taken private or restructured
into insolvency and have no surviving Yahoo Finance history at any ticker
variant tried; their addition events are also dropped.

The resulting ``NAME_TO_TICKER`` mapping was hand-verified against Yahoo
Finance (including cross-exchange fallbacks — e.g. Covestro's Xetra listing
serves no historical Yahoo data, so it's priced off its Frankfurt listing
instead) rather than guessed from company-name conventions.
"""

from __future__ import annotations

import datetime as dt
import os
import re

PDF_URL = "https://www.stoxx.com/document/Indices/Common/Indexguide/Historical_Index_Compositions.pdf"

# (STOXX section header substring, benchmark index Yahoo ticker)
INDEX_CONFIG = {
    "DAX": ("DAX® INDEX COMPOSITION", "^GDAXI"),
    "TecDAX": ("TECDAX® INDEX COMPOSITION", "^TECDAX"),
}

DATE = r"(\d{2})\.(\d{2})\.(\d{4})"
_LINE_RE = re.compile(rf"^{DATE}\s+{DATE}\s+(.*)$")
_DATE_ONLY_RE = re.compile(rf"^{DATE}\s+{DATE}\s*$")

# Company name -> Yahoo ticker, hand-verified (see module docstring). Only
# covers names actually appearing in the DAX/TecDAX tables since 2018.
NAME_TO_TICKER: dict[str, str] = {
    "ProSiebenSat.1 Media SA": "PSM.DE", "Covestro AG": "1COV.F", "Covestro": "1COV.F",
    "Commerzbank AG": "CBK.DE", "Wirecard AG": "WDI.HM",
    "Thyssenkrupp AG": "TKA.DE", "MTU Aero Engines": "MTX.DE",
    "Deutsche Lufthansa AG": "LHA.DE", "Deutsche Wohnen SE": "DWNI.DE",
    "Delivery Hero SE": "DHER.DE",
    "Beiersdorf AG": "BEI.DE", "Siemens Energy AG": "ENR.DE",
    "Airbus SE": "AIR.DE", "Brenntag SE": "BNR.DE", "HelloFresh SE": "HFG.DE",
    "Porsche Automobile Holding VZO": "PAH3.DE", "Puma SE": "PUM.DE", "Qiagen NV": "QIA.DE",
    "Sartorius AG VZ": "SRT3.DE", "Siemens Healthineers AG": "SHL.DE",
    "Symrise AG": "SY1.DE", "Zalando SE": "ZAL.DE",
    "Daimler Truck Holding": "DTG.DE", "Hannover Rück SE": "HNR1.DE",
    "Dr. Ing. h.c. F. Porsche VZOI": "P911.DE",
    "Linde PLC": "LIN.DE", "Fresenius Medical Care": "FME.DE", "Rheinmetall AG": "RHM.DE",
    "Fresenius Medical Care AG": "FME.DE",
    "GEA Group": "G1A.DE", "Scout24": "G24.DE", "Hochtief": "HOT.DE",
    "Porsche Automobil Holding": "PAH3.DE", "Pref Hochtief": "HOT.DE",
    # TecDAX
    "ADVA Optical Networking SE": "ADV.DE", "Aumann AG": "AAG.DE",
    "GFT Technologies SE": "GFT.DE",
    # ISRA VISION AG / Isra Vision: taken private (Atlas Copco, 2020); no
    # surviving Yahoo history under any ticker variant tried -- not mapped.
    "Medigene AG": "MDG1.DE", "SAP SE": "SAP.DE",
    "SLM Solutions AG": "AM3D.DE", "Deutsche Telekom AG": "DTE.DE",
    "SMA Solar Technol.": "S92.DE", "Infineon Tech. AG": "IFX.DE",
    "Draegerwerk": "DRW3.DE",
    # Varta AG: 2024 insolvency restructuring, shares cancelled -- not mapped.
    "TeamViewer AG": "TMV.DE",
    "Dialog Semiconductor": "DLG.DE",
    "RIB Software": "RSTA.DE",
    "Eckert + Ziegler": "EUZ.DE", "Eckert+Ziegler AG": "EUZ.DE", "Eckert+Ziegler": "EUZ.DE",
    "LPKF Laser & Electronics AG": "LPK.DE", "LPKF Laser+Electronics": "LPK.DE",
    "New Work SE": "NWO.DE", "SMA Solar Technology AG": "S92.DE",
    # SUSE S.A.: taken private (EQT, 2024); no surviving Yahoo history -- not mapped.
    "Vantage Towers AG": "VTWR.HM",
    "Pfeiffer Vacuum Tech.": "PFV.DE", "Nagarro SE": "NA9.DE", "Nagarro": "NA9.DE",
    "Hensoldt AG": "HAG.DE", "SMA Solar Technology": "S92.DE",
    "1&1 AG": "1U1.DE", "1&1": "1U1.DE", "Adtran Holdings": "ADV.DE", "Nordex SE": "NDX1.DE",
    "Atoss Software AG": "AOF.DE",
    "Evotec SE": "EVT.DE",
    "Verbio Ver.Bioenergie": "VBK.DE", "Verbio": "VBK.DE",
    "Kontron AG": "KTN.DE", "Software AG": "SOW.DE", "PNE AG": "PNE3.DE", "PNE": "PNE3.DE",
    "Telefonica": "O2D.DE", "Suess Microtec": "SMHN.DE",
    "MorphoSys": "MOR.DE", "Elmos Semiconductor": "ELG.DE",
    "IONOS Group": "IOS.DE", "Nexus": "NXU.DE",
    "Draegerwerk Pref": "DRW3.DE",
    "CompuGroup Medical": "COP.DE",
    "PVA Tepla": "TPE.DE",
    "Energiekontor": "EKT.DE",
    "Formycon": "FYB.DE", "Ottobock": "OBCK.DE",
}


def fetch_stoxx_pdf(cache_path: str | None = None) -> bytes:
    """Download the STOXX historical-compositions PDF (cached to disk if given)."""
    import subprocess

    if cache_path and os.path.exists(cache_path):
        return open(cache_path, "rb").read()
    ca = "/root/.ccr/ca-bundle.crt"
    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    args = ["curl", "-sSL", "-H", f"User-Agent: {ua}"]
    if os.path.exists(ca):
        args += ["--cacert", ca]
    args.append(PDF_URL)
    data = subprocess.run(args, capture_output=True, timeout=60).stdout
    if cache_path:
        with open(cache_path, "wb") as fh:
            fh.write(data)
    return data


def _pdf_page_texts(pdf_bytes: bytes) -> list[str]:
    import io

    import pypdf  # lazy: only needed when parsing a freshly-downloaded PDF

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return [p.extract_text() for p in reader.pages]


def section_text(pdf_bytes: bytes, section_marker: str) -> str:
    """Concatenate every page belonging to one index's section.

    STOXX's PDF repeats a running header on every page but only marks the
    *first* page of each index's table with an inline ``N.  NAME INDEX
    COMPOSITION`` title — so sections are found by tracking the most recent
    marker seen, not by a single grep match (which would find the table of
    contents entry, not the real section boundary).
    """
    marker_re = re.compile(r"^\d+\.\s+([A-Z][A-Z0-9® ]+INDEX COMPOSITION)", re.M)
    current = None
    pages: list[str] = []
    for text in _pdf_page_texts(pdf_bytes):
        m = marker_re.search(text)
        if m:
            current = m.group(1).strip()
        if current == section_marker:
            pages.append(text)
    return "\n".join(pages)


def _clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if re.match(r"^\d+/\d+$", s):
            continue
        if "HISTORICAL INDEX COMPOSITIONS" in s:
            continue
        if re.match(r"^Date of$|^change$|^announcement.*Deletion|^Deletion|^Addition", s):
            continue
        if "announcement Deletion" in s:
            continue
        if s.startswith("*"):
            continue
        if re.match(r"^\d+\.\s+[A-Z]", s) and "INDEX COMPOSITION" in s:
            continue
        lines.append(s)
    return lines


def parse_change_blocks(text: str, min_year: int = 2018):
    """Group cleaned lines into ``(effective, announcement, raw_pairs)`` blocks.

    Each block is one reconstitution date; ``raw_pairs`` is the list of
    deletion/addition text fragments for that date (usually one "Deletion
    Addition" string per swap, sometimes several for a multi-company event).
    """
    lines = _clean_lines(text)
    blocks, i = [], 0
    while i < len(lines):
        s = lines[i]
        m = _LINE_RE.match(s)
        m0 = _DATE_ONLY_RE.match(s)
        if not (m or m0):
            i += 1
            continue
        if m:
            d1, mo1, y1, d2, mo2, y2, rest = m.groups()
            pairs = [rest]
        else:
            d1, mo1, y1, d2, mo2, y2 = m0.groups()
            pairs = []
        eff = dt.date(int(y1), int(mo1), int(d1))
        ann = dt.date(int(y2), int(mo2), int(d2))
        i += 1
        while i < len(lines) and not _LINE_RE.match(lines[i]) and not _DATE_ONLY_RE.match(lines[i]):
            pairs.append(lines[i])
            i += 1
        blocks.append((eff, ann, pairs))
    return [b for b in blocks if b[0].year >= min_year]


def _find_known_names(text: str, names: dict[str, str]):
    """Every known company name found in ``text``, as (position, name), sorted
    by position. Longest names are matched first so e.g. "Eckert+Ziegler AG"
    wins over the shorter "Eckert+Ziegler"."""
    hits = []
    for name in sorted(names, key=len, reverse=True):
        idx = text.find(name)
        if idx != -1 and not any(idx >= h[0] and idx < h[0] + len(h[1]) for h in hits):
            hits.append((idx, name))
    return sorted(hits)


def extract_additions(text: str, min_year: int = 2018, name_to_ticker: dict | None = None):
    """Turn parsed change blocks into ``(effective, announcement, ticker)``
    addition events, using the known-name dictionary to resolve which company
    in each block is the *addition* (the STOXX table's right-hand column).

    Blocks explicitly marked as pure-addition (deletion column is a literal
    ``-``, e.g. the 2021 DAX 30->40 expansion) treat every remaining line as
    an addition. Otherwise a line must contain exactly two recognized names
    (deletion, addition) to be resolved unambiguously; anything else is
    returned in ``dropped`` rather than guessed.
    """
    names = name_to_ticker if name_to_ticker is not None else NAME_TO_TICKER
    additions, dropped = [], []
    for eff, ann, pairs in parse_change_blocks(text, min_year):
        if pairs and pairs[0].strip() == "-":
            for p in pairs[1:]:
                hits = _find_known_names(p, names)
                if len(hits) == 1 and hits[0][1] == p.strip():
                    additions.append((eff, ann, names[hits[0][1]]))
                else:
                    dropped.append((eff, ann, p, "unmatched pure-addition line"))
            continue
        for p in pairs:
            hits = _find_known_names(p, names)
            if len(hits) == 2:
                additions.append((eff, ann, names[hits[1][1]]))  # 2nd = addition column
            else:
                dropped.append((eff, ann, p, f"{len(hits)} known names matched"))
    return additions, dropped


def load_events(index: str, pdf_cache: str | None = None, min_year: int = 2018):
    """Fetch (or read cached) the STOXX PDF and return that index's addition
    events as ``(effective, announcement, ticker)`` tuples."""
    marker, _ = INDEX_CONFIG[index]
    pdf_bytes = fetch_stoxx_pdf(pdf_cache)
    text = section_text(pdf_bytes, marker)
    additions, _dropped = extract_additions(text, min_year)
    return additions
