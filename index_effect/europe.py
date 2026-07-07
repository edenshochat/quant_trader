"""European index-inclusion events: DAX, TecDAX, MDAX, and SDAX (Deutsche Börse
/ STOXX) — the full German blue-chip-to-small-cap ladder, paralleling the US
S&P 500/400/600 + Nasdaq-100 coverage.

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
Wikipedia tables. This module parses all four sections.

A meaningful fraction of MDAX/SDAX rows couldn't be safely resolved: the PDF's
column layout is sometimes lost in text extraction (no way to tell which name
was added vs. removed on a jumbled multi-company reconstitution date), and
some long company names wrap across lines in a way this parser doesn't
currently rejoin (e.g. "Steinhoff International" / "Holdings NV" as two
fragments) — both cases are dropped rather than guessed, see
``extract_additions``'s ``dropped`` return value. This lowers MDAX/SDAX's
yield noticeably versus DAX/TecDAX (where most reconstitutions are clean
single swaps); rejoining wrapped names is a documented follow-up, not
attempted here. Companies taken private, merged away, or restructured into
insolvency with no surviving Yahoo Finance history at any ticker variant
tried (e.g. ISRA VISION, Varta, SUSE S.A., Steinhoff International, Gerry
Weber, Leoni, Synlab, Vitesco Technologies, About You Holding, comdirect
bank) are also dropped, as are a handful of resolved tickers (Metro AG,
CompuGroup Medical, Dialog Semiconductor, Shop Apotheke Europe, Software AG)
whose only surviving Yahoo listing is a regional German exchange (Hamburg,
Munich, Frankfurt) with historical depth that doesn't reach back to their
particular addition dates — a real symbol, but Yahoo's archive for it is
too shallow for that specific event.

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
    "MDAX": ("MDAX® INDEX COMPOSITION", "^MDAXI"),
    "SDAX": ("SDAX® INDEX COMPOSITION", "^SDAXI"),
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
    "Dialog Semiconductor": "DLGS.DE",
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
    "Kontron AG": "KTN.DE", "Software AG": "SOW.F", "PNE AG": "PNE3.DE", "PNE": "PNE3.DE",
    "Telefonica": "O2D.DE", "Suess Microtec": "SMHN.DE",
    "MorphoSys": "MOR.DE", "Elmos Semiconductor": "ELG.DE",
    "IONOS Group": "IOS.DE", "Nexus": "NXU.DE",
    "Draegerwerk Pref": "DRW3.DE",
    "CompuGroup Medical": "COP.HM",
    "PVA Tepla": "TPE.DE",
    "Energiekontor": "EKT.DE",
    "Formycon": "FYB.DE", "Ottobock": "OBCK.DE",
    # MDAX
    "Ceconomy AG": "CEC.DE", "Jungheinrich AG VZO": "JUN3.DE", "Qiagen": "QIA.DE",
    # Leoni AG: no surviving Yahoo history found under any ticker tried -- not mapped.
    "Siemens Health.": "SHL.DE",
    "Stroeer SE + CO. KGAA": "SAX.DE", "UTD. Internet AG": "1U1.DE",
    "Talanx AG": "TLX.DE", "Sartorius AG VZO": "SRT3.DE",
    "Morphosys AG": "MOR.DE", "Freenet AG": "FNTN.DE", "Siltronic AG": "WAF.DE",
    "Nemetschek SE": "NEM.DE", "Bechtle AG": "BC8.DE", "Alstria Office REIT-AG": "AOX.DE",
    "CTS Eventim": "EVD.DE", "Carl Zeiss Meditec AG": "AFX.DE",
    "Knorr-Bremse AG": "KBX.DE", "Salzgitter AG": "SZG.DE", "Schaeffler AG": "SHA0.DE",
    "Wacker Chemie AG": "WCH.DE", "Grenke AG": "GLJ.DE", "Axel Springer SE": "SPR.DE",
    "Cancom SE": "COK.DE", "Innogy SE": "IGY.DE",
    "Deutsche EuroShop": "DEQ.DE", "Norma Group SE": "NOEJ.DE", "Rational AG": "RAA.DE",
    "Fielmann AG": "FIE.DE", "Deutsche Pfandbriefbank AG": "PBB.DE",
    "RTL Group": "RRTL.DE", "Shop Apotheke Europe": "SAE1.MU", "Rocket Internet": "RKET.HM",
    "Aareal Bank AG": "ARL.DE", "Metro AG": "B4B.HM", "Encavis AG": "CAP.DE",
    "Osram Licht AG": "OSR.HM", "Auto1 Group SE": "AG1.DE", "Befesa S.A.": "BFSA.DE",
    "Hypoport SE": "HYQ.DE", "Zooplus AG": "ZO1.HM",
    "Hella GmbH + Co. KGAA": "HLE.DE", "Sixt SE": "SIX2.DE", "Uniper SE": "UN0.DE",
    "Grand City Properties": "GYC.DE", "Stabilus SE": "STM.F",
    "Jenoptik AG": "JEN.DE", "Krones AG": "KRN.DE", "Redcare Pharmacy": "RDC.DE",
    "Aroundtown SA": "AT1.DE",
    # Vitesco Technologies Group: no surviving Yahoo history found -- not mapped.
    "Prosiebensat.1": "PSM.DE", "Duerr": "DUE.DE", "Bilfinger": "GBF.DE",
    "SIXT": "SIX2.DE", "TUI": "TUI1.DE", "Traton": "8TRA.DE", "Schott Pharma": "1SXP.DE",
    "Gerresheimer": "GXI.DE", "DWS Group GmbH & Co. KgaA": "DWS.DE",
    "FlatexDEGIRO N": "FTK.DE", "Renk": "R3NK.DE", "Deutz": "DEZ.DE",
    "Compugroup Med.": "COP.HM", "Hannover Rueck": "HNR1.DE",
    "Uniper": "UN01.DE", "Verbio Ver. Bionenergie": "VBK.DE",
    "Rheinmetall AG": "RHM.DE", "Verbio AG": "VBK.DE",
    "UTD. Internet AG Evotec SE": "EVT.DE",  # (defensive alias; see dropped-line notes)
    "Vantage Towers AG Hochtief AG": "HOT.DE",
    # SDAX
    "Südzucker AG": "SZU.DE", "CORESTATE Capital Holding S.A.": "CCAP.DE",
    "JOST Werke AG": "JST.DE", "Scout24 AG": "G24.DE", "Delivery Hero AG": "DHER.DE",
    "DWS Group GmbH & Co. KgaA ": "DWS.DE", "Aumann AG": "AAG.DE",
    "Biotest AG VZ": "BIO3.DE", "Elringklinger AG": "ZIL2.DE", "Grammer AG": "GMM.DE",
    "Pfeiffer Vacuum Tech.": "PFV.DE", "RIB Software SE": "RSTA.DE",
    "Dr. Hoenle AG": "HNL.DE", "Dr. Hönle AG": "HNL.DE", "BayWa AG": "BYW6.DE",
    "MediGene AG": "MDG1.DE", "Amadeus FiRe AG": "AAD.DE", "DMG Mori AG": "GIL.DE",
    "VTG AG": "VT9.DE", "Vossloh AG": "VOS.DE", "Eckert & Ziegler AG": "EUZ.DE",
    "Hapag-Lloyd AG": "HLAG.DE", "Instone Real Estate Group AG": "INS.DE",
    "Dermapharm Holding": "DMP.DE",
    # comdirect bank AG: absorbed into Commerzbank (2020); no surviving Yahoo history -- not mapped.
    "TLG Immobilien AG": "TLG.DE", "LPKF Laser & Electronics AG": "LPK.DE",
    "Adler Real Estate AG": "ADL.DE", "Elmos Semiconductor AG": "ELG.DE",
    "Godewind Immobilien AG": "GWD.DE", "SGL Carbon": "SGL.DE",
    "SNP Schneider-Neureither &": "SHF.DE", "Heidelberger": "HDD.DE",
    "Sixt Leasing": "SIX3.DE", "MLP SE": "MLP.DE", "Rhön-Klinikum AG": "RHK.DE",
    "Tele Columbus AG": "TC1.HM", "HORNBACH-Baumarkt-AG": "HBM.DE",
    "Bertrandt AG": "BDT.DE", "Medios AG": "ILM1.DE", "Secunet Security Networks": "YSN.DE",
    "CropEnergies AG": "CE2.HM", "FlatexDEGIRO AG": "FTK.DE", "WashTec AG": "WSU.DE",
    "Deutsche Beteiligungs AG": "DBAN.DE", "Koenig + Bauer AG": "SKB.DE",
    "Corestate Capital": "CCAP.DE", "Nordex SE": "NDX1.DE",
    # About You Holding: no surviving Yahoo history found under any ticker tried -- not mapped.
    "Borussia Dortmund": "BVB.DE", "PVA Tepla AG": "TPE.DE",
    "Sto SE+Co. KGAA VZO": "STO3.DE", "Basler AG": "BSL.DE",
    # Synlab AG: no surviving Yahoo history found under any ticker tried -- not mapped.
    "Basler": "BSL.DE", "Adesso SE": "ADN1.DE", "Adler Group S.A.": "ADJ.DE",
    "Takkt AG": "TTK.DE", "SFC Energy AG": "F3C.DE", "DIC Asset AG": "DIC.DE",
    "Mutares SE & Co.": "MUX.DE", "KSB": "KSB3.DE", "ZEAL Network SE": "TIMA.DE",
    "Thyssenkrupp Nucera": "NCH2.DE", "Alzchem Group AG": "ACT.DE",
    "Douglas": "DOU.DE", "Nexus": "NXU.DE", "Springer Nature": "SPG1.DE",
    "New Work SE": "NWO.DE", "Instone Real Estate Group": "INS.DE",
    "Adtran": "ADV.DE", "AUTO1 Group": "AG1.DE",
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
