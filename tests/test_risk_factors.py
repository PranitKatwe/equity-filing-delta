"""Tests for the LLM classifier's feature aggregation — mocked, no API call.

We inject a fake OpenAI-shaped client so the logic that turns per-passage
classifications into event features is tested without a key or network.
"""

import json

from eqd.delta.diff import SectionDiff
from eqd.delta.risk_factors import classify_added, classify_diff


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, text):
        self._text = text
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _Resp(self._text)


class _Chat:
    def __init__(self, text):
        self.completions = _Completions(text)


class _Client:
    def __init__(self, payload, *, raw: str | None = None):
        self.chat = _Chat(raw if raw is not None else json.dumps(payload))


def test_classify_diff_aggregates_categories():
    diff = SectionDiff(added=["aaa", "bbb", "ccc"])
    payload = {
        "classifications": [
            {"index": 0, "category": "new_substantive_risk", "rationale": "new AI risk"},
            {"index": 1, "category": "boilerplate_or_reorder", "rationale": "generic intro"},
            {"index": 2, "category": "expanded_existing_risk", "rationale": "more detail"},
        ]
    }
    feats = classify_diff(diff, client=_Client(payload))
    assert feats["n_new_substantive_risk"] == 1
    assert feats["n_substantive_added"] == 2   # new + expanded
    assert feats["n_boilerplate_added"] == 1


def test_empty_added_makes_no_api_call():
    client = _Client({"classifications": []})
    feats = classify_diff(SectionDiff(added=[]), client=client)
    assert feats["n_substantive_added"] == 0
    assert client.chat.completions.calls == 0   # short-circuits before the API


def test_classify_added_returns_rows():
    payload = {"classifications": [{"index": 0, "category": "reworded_same_meaning", "rationale": "r"}]}
    rows = classify_added(["only one"], client=_Client(payload))
    assert rows == [{"index": 0, "category": "reworded_same_meaning", "rationale": "r"}]


def test_classifier_tolerates_think_trace_and_bad_rows():
    # A reasoning trace plus one malformed row: the trace is stripped, the bad row dropped.
    raw = (
        "<think>reasoning here { not json }</think>\n"
        + json.dumps({"classifications": [
            {"index": 0, "category": "new_substantive_risk", "rationale": "ok"},
            {"index": 1, "category": "not_a_category", "rationale": "bad"},
        ]})
    )
    rows = classify_added(["a", "b"], client=_Client({}, raw=raw))
    assert rows == [{"index": 0, "category": "new_substantive_risk", "rationale": "ok"}]
