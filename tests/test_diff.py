"""Tests for the mechanical diff — the layer that keeps LLM extraction honest.

Synthetic risk-factor sets with a known change structure: one unchanged, one
reworded, one removed, one added. The diff must recover exactly that.
"""

from eqd.delta.diff import diff_text, split_sentences

# Each "risk factor" is one paragraph, made long enough to clear _MIN_PARA_CHARS.
RF_UNCHANGED = (
    "Our business is highly dependent on global supply chains, and disruptions "
    "to those chains could materially harm our results of operations."
)
RF_REMOVED = (
    "We face significant risks related to the ongoing COVID-19 pandemic, which "
    "has disrupted manufacturing and reduced consumer demand across regions."
)
RF_PRIOR_MODIFIED = (
    "Cybersecurity incidents could compromise our systems and confidential data, "
    "exposing us to litigation and reputational harm that could be significant."
)
RF_CURRENT_MODIFIED = (
    "Cybersecurity incidents, including those involving artificial intelligence, "
    "could compromise our systems and confidential data, exposing us to "
    "litigation and reputational harm that could be significant."
)
RF_ADDED = (
    "New and evolving regulations governing artificial intelligence may increase "
    "our compliance costs and restrict how we develop and deploy new products."
)

PRIOR = "\n".join([RF_UNCHANGED, RF_REMOVED, RF_PRIOR_MODIFIED])
CURRENT = "\n".join([RF_UNCHANGED, RF_PRIOR_MODIFIED.replace(RF_PRIOR_MODIFIED, RF_CURRENT_MODIFIED), RF_ADDED])


def test_split_sentences_splits_and_drops_fragments():
    text = RF_UNCHANGED + "\n" + RF_REMOVED + "\n42"
    sents = split_sentences(text)
    assert len(sents) == 2                # "42" fragment dropped
    assert sents[0] == RF_UNCHANGED
    assert sents[1] == RF_REMOVED


def test_diff_recovers_known_change_structure():
    d = diff_text(PRIOR, CURRENT)
    s = d.summary()

    # Exactly one genuinely new passage, and it's the AI-regulation risk.
    assert s["n_added"] == 1
    assert "artificial intelligence" in d.added[0].lower()

    # Exactly one removed passage, and it's the COVID risk.
    assert s["n_removed"] == 1
    assert "covid" in d.removed[0].lower()

    # The cybersecurity risk was reworded, not added/removed.
    assert s["n_modified"] == 1
    prior_m, current_m, ratio = d.modified[0]
    assert "cybersecurity" in prior_m.lower()
    assert ratio >= 0.60

    # The supply-chain risk is untouched.
    assert s["n_unchanged"] == 1


def test_identical_sections_show_no_changes():
    d = diff_text(PRIOR, PRIOR)
    assert d.summary() == {
        "n_added": 0,
        "n_removed": 0,
        "n_modified": 0,
        "n_unchanged": 3,
        "doc_similarity": 1.0,
    }
