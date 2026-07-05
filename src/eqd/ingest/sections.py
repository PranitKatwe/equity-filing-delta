"""sections.py — pull the text blocks we diff out of filing HTML.

Targets:
  - 10-K Item 1A "Risk Factors"  (the canonical Lazy-Prices signal)
  - 10-K Item 7  "MD&A"          (guidance / forward-language)

SEC filings are messy inline-XBRL HTML with a table of contents that repeats
every item header, plus in-body cross-references ("see Item 7"). We locate a
section by taking the span from a candidate item header to its bounding next
item, and keeping the candidate with the most text — the real body dwarfs a
one-line TOC entry or a cross-ref. Not perfect across 500 filers, but robust
enough; extraction quality is measured on a gold set in Step 3.
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# SEC-mandated section titles, used to anchor REAL headings (both the target
# sections and their end bounds). Anchoring ends on titles too means an in-body
# cross-reference like "...read with Item 8" is NOT mistaken for the boundary —
# only the true "Item 8. Financial Statements..." heading is.
_ITEM_TITLES = {
    "1A": r"Risk\s+Factors",
    "1B": r"Unresolved\s+Staff\s+Comments",
    "2": r"Properties",
    "7": r"Management.{0,3}s\s+Discussion\s+and\s+Analysis",
    "7A": r"Quantitative\s+and\s+Qualitative\s+Disclosures?\s+About\s+Market\s+Risk",
    "8": r"Financial\s+Statements\s+and\s+Supplementary\s+Data",
}

# Per target section: output key and the item(s) whose heading ends it.
_SECTIONS = {
    "1A": {"key": "item_1a", "ends": ["1B", "2"]},
    "7": {"key": "item_7", "ends": ["7A", "8"]},
}
_MIN_SECTION_CHARS = 1000  # below this, we almost certainly grabbed a TOC line
# Chars of punctuation/whitespace tolerated between the item id and its title.
_HEADING_GAP = r"[\.\s:\-–—]*"


def html_to_text(html: str) -> str:
    """Flatten filing HTML to normalized plain text (newline-separated blocks)."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _item_starts(text: str, item: str) -> list[int]:
    """Positions of bare 'Item <n>' markers (used for END bounds only)."""
    # Word boundary after the item id so 'Item 7' doesn't match 'Item 7A'.
    pat = rf"item\s*{re.escape(item)}(?![0-9A-Za-z])"
    return [m.start() for m in re.finditer(pat, text, re.I)]


def _heading_starts(text: str, item: str) -> list[int]:
    """Positions of TRUE section headings: 'Item <n>' directly followed by its
    title (e.g. 'Item 1A. Risk Factors'). Excludes cross-refs like
    'Item 1A of this Form 10-K under the heading Risk Factors'. Falls back to
    the bare 'Item <n>' marker if the item has no registered title."""
    title = _ITEM_TITLES.get(item)
    if not title:
        return _item_starts(text, item)
    pat = rf"item\s*{re.escape(item)}{_HEADING_GAP}{title}"
    hits = [m.start() for m in re.finditer(pat, text, re.I)]
    return hits or _item_starts(text, item)  # fallback if title phrasing differs


def extract_section(text: str, item: str) -> str | None:
    """Best-effort text of one item section, or None if not confidently found."""
    spec = _SECTIONS[item]
    starts = _heading_starts(text, item)
    end_positions = sorted(p for e in spec["ends"] for p in _heading_starts(text, e))

    best = ""
    for s in starts:
        after = [p for p in end_positions if p > s]  # nearest bound after start
        if not after:
            continue
        span = text[s : after[0]]
        if len(span) > len(best):
            best = span

    return best if len(best) >= _MIN_SECTION_CHARS else None


def extract_sections(html: str, form: str = "10-K") -> dict[str, str | None]:
    """Return {'item_1a': ..., 'item_7': ...} text for a 10-K.

    (10-Q handling — Part I Item 2 MD&A, Part II Item 1A updates — is added
    when we extend to quarterly deltas; 10-K risk factors are the M1 focus.)
    """
    text = html_to_text(html)
    return {spec["key"]: extract_section(text, item) for item, spec in _SECTIONS.items()}
