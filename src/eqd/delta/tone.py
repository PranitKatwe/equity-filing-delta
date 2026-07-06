"""delta/tone.py — Loughran-McDonald tone deltas (transparent, non-LLM baseline).

LM proportion of a text = (# tokens in a sentiment category) / (# total tokens).
The YoY tone delta = current proportion - prior proportion. Rising negative or
uncertainty tone is the transparent analog of "worse risk language" and
cross-checks the (later) LLM tone score — if the two disagree wildly, trust the
transparent one and investigate (DESIGN §5).

Requires the cached word lists from scripts/fetch_lm_dictionary.py.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..config import DATA

LM_DIR = DATA / "lm"
CATEGORIES = ("negative", "uncertainty", "litigious", "positive", "constraining")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*")


@lru_cache(maxsize=8)
def load_wordset(category: str) -> frozenset[str]:
    path = LM_DIR / f"{category}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/fetch_lm_dictionary.py first."
        )
    return frozenset(w.strip().upper() for w in path.read_text(encoding="utf-8").splitlines() if w.strip())


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.upper())


def proportions(text: str) -> dict[str, float]:
    """Fraction of tokens falling in each LM category."""
    toks = tokenize(text)
    n = len(toks) or 1
    return {c: sum(t in load_wordset(c) for t in toks) / n for c in CATEGORIES}


def tone_features(prior_text: str, current_text: str) -> dict[str, float]:
    """Current tone level + YoY tone delta for each category.

    Keys: tone_<cat> (current proportion) and d_tone_<cat> (current - prior).
    """
    pri = proportions(prior_text)
    cur = proportions(current_text)
    out: dict[str, float] = {}
    for c in CATEGORIES:
        out[f"tone_{c}"] = round(cur[c], 6)
        out[f"d_tone_{c}"] = round(cur[c] - pri[c], 6)
    return out
