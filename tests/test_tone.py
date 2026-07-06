"""Tests for the Loughran-McDonald tone baseline.

Skipped when the cached word lists are absent (data/ is gitignored; run
scripts/fetch_lm_dictionary.py to populate them).
"""

import pytest

from eqd.delta.tone import CATEGORIES, load_wordset, proportions, tone_features

try:
    load_wordset("negative")
    _HAVE_LM = True
except FileNotFoundError:
    _HAVE_LM = False

pytestmark = pytest.mark.skipif(not _HAVE_LM, reason="LM word lists not fetched")

STABLE = "The company is stable, profitable, and continues to grow its dividend."
GRIM = (
    "The company faces litigation, severe losses, adverse regulatory actions, "
    "default, impairment, termination, and bankruptcy risks that could be material."
)


def test_negative_proportion_rises_with_grim_text():
    assert proportions(GRIM)["negative"] > proportions(STABLE)["negative"]


def test_tone_features_keys_and_delta_sign():
    feats = tone_features(prior_text=STABLE, current_text=GRIM)
    for c in CATEGORIES:
        assert f"tone_{c}" in feats and f"d_tone_{c}" in feats
    # Going from stable -> grim, negative tone delta must be positive.
    assert feats["d_tone_negative"] > 0


def test_empty_text_is_safe():
    p = proportions("")
    assert all(v == 0.0 for v in p.values())
