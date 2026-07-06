"""diff.py — mechanically isolate what changed between two filings.

This is the honesty layer. We diff the prior and current section text and
surface concrete changed passages; the LLM (risk_factors.py) only ever
classifies passages produced here, so it cannot invent changes and every
feature traces back to source text.

Method: split each section into *sentences* (after flattening line breaks) and
align them with difflib. We diff sentences rather than paragraphs because HTML
paragraph chunking is unstable year over year — the same risk factor may be one
big block in one filing and several small ones the next — which fabricates
spurious add/remove churn. Sentence boundaries (periods) are stable regardless
of chunking, so the diff reflects real edits. Insert/delete blocks and
similarity-reconciled replaces yield *added*, *removed*, and *modified* units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from rapidfuzz import fuzz, process

_MIN_UNIT_CHARS = 30          # drop fragments (page numbers, stray headers)
_MODIFIED_THRESHOLD = 0.60    # matched sentences this similar count as reworded
_IDENTICAL_THRESHOLD = 0.95   # this similar == the same sentence, merely moved

# Split on sentence-ending punctuation followed by whitespace + a capital/quote/
# digit. Imperfect around abbreviations, but the same way every year, so the
# diff still aligns.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(])')


def split_sentences(text: str) -> list[str]:
    """Section text -> list of sentences worth diffing (chunking-independent)."""
    flat = " ".join(text.split())  # collapse all line breaks / whitespace first
    sents = _SENT_SPLIT.split(flat)
    return [s.strip() for s in sents if len(s.strip()) >= _MIN_UNIT_CHARS]


def _norm(p: str) -> str:
    """Whitespace/case-insensitive key so trivially-identical paras align as equal."""
    return " ".join(p.lower().split())


@dataclass
class SectionDiff:
    """Structured, provenance-bearing result of diffing one section."""

    added: list[str] = field(default_factory=list)       # new passages (candidate new risks)
    removed: list[str] = field(default_factory=list)      # dropped passages
    modified: list[tuple[str, str, float]] = field(       # (prior, current, similarity)
        default_factory=list
    )
    n_unchanged: int = 0
    doc_similarity: float = 1.0                            # paragraph-level overall similarity

    def summary(self) -> dict:
        return {
            "n_added": len(self.added),
            "n_removed": len(self.removed),
            "n_modified": len(self.modified),
            "n_unchanged": self.n_unchanged,
            "doc_similarity": round(self.doc_similarity, 4),
        }


def diff_text(
    prior: str,
    current: str,
    *,
    modified_threshold: float = _MODIFIED_THRESHOLD,
    identical_threshold: float = _IDENTICAL_THRESHOLD,
) -> SectionDiff:
    """Diff two versions of a section's text into added/removed/modified passages.

    Two passes:
      1. Positional diff (difflib) -> equal blocks, plus raw candidate added
         (inserts + replace-currents) and removed (deletes + replace-priors).
      2. Global reconciliation -> match each raw-added passage to its most
         similar raw-removed passage anywhere in the section, so a paragraph
         that merely moved or was lightly edited is NOT double-counted as one
         addition and one removal (the boilerplate-reordering trap).
    """
    p = split_sentences(prior)
    c = split_sentences(current)
    sm = SequenceMatcher(a=[_norm(x) for x in p], b=[_norm(x) for x in c], autojunk=False)

    raw_added: list[str] = []
    raw_removed: list[str] = []
    out = SectionDiff(doc_similarity=sm.ratio())
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.n_unchanged += i2 - i1
        elif tag == "insert":
            raw_added.extend(c[j1:j2])
        elif tag == "delete":
            raw_removed.extend(p[i1:i2])
        elif tag == "replace":
            raw_added.extend(c[j1:j2])
            raw_removed.extend(p[i1:i2])

    _reconcile(raw_added, raw_removed, out, modified_threshold, identical_threshold)
    return out


def _reconcile(
    raw_added: list[str],
    raw_removed: list[str],
    out: SectionDiff,
    modified_threshold: float,
    identical_threshold: float,
) -> None:
    """Match raw-added to raw-removed passages by similarity (rapidfuzz, C-speed).

    fuzz.ratio returns 0..100; we scale to 0..1. A precomputed similarity matrix
    (process.cdist) makes this ~1000x faster than pairwise difflib, which
    otherwise dominates panel-build time.
    """
    if not raw_added or not raw_removed:
        out.added = list(raw_added)
        out.removed = list(raw_removed)
        return

    nadd = [_norm(a) for a in raw_added]
    nrem = [_norm(r) for r in raw_removed]
    scores = process.cdist(nadd, nrem, scorer=fuzz.ratio)  # shape (A, R), 0..100

    used: set[int] = set()
    for ai, cur in enumerate(raw_added):
        row = scores[ai]
        best_i, best = -1, -1.0
        for ri in range(len(raw_removed)):
            if ri not in used and row[ri] > best:
                best, best_i = float(row[ri]), ri
        ratio = best / 100.0
        if best_i >= 0 and ratio >= identical_threshold:
            used.add(best_i)
            out.n_unchanged += 1  # same passage, merely relocated
        elif best_i >= 0 and ratio >= modified_threshold:
            used.add(best_i)
            out.modified.append((raw_removed[best_i], cur, round(ratio, 4)))
        else:
            out.added.append(cur)
    out.removed = [pri for i, pri in enumerate(raw_removed) if i not in used]
