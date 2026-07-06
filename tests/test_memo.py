"""Tests for the grounded narrator — mocked, no API call.

The key property to guard: the narrator is only ever handed whitelisted,
pre-computed result fields. It cannot be fed arbitrary/future data, and the
system prompt forbids advice.
"""

from eqd.narrate.memo import event_memo


class _Block:
    type = "text"

    def __init__(self, t):
        self.text = t


class _Resp:
    def __init__(self, t):
        self.content = [_Block(t)]


class _Msgs:
    def __init__(self, text):
        self.text = text
        self.last = None

    def create(self, **kwargs):
        self.last = kwargs
        return _Resp(self.text)


class _Client:
    def __init__(self, text):
        self.messages = _Msgs(text)


def test_event_memo_is_grounded_to_whitelisted_fields():
    row = {
        "ticker": "AAPL",
        "accession": "0000320193-25-000079",
        "net_added": 5,
        "market_model__car_0_5": 0.021,
        "leaked_future_return": 0.99,   # not a whitelisted field — must not be sent
    }
    client = _Client("AAPL's Item 1A added 5 net risk sentences (0000320193-25-000079).")
    out = event_memo(row, client=client)

    payload = client.messages.last["messages"][0]["content"]
    assert "net_added" in payload and "accession" in payload   # grounded facts included
    assert "leaked_future_return" not in payload               # non-whitelisted excluded
    assert "advice" not in client.messages.last["system"].lower() or "no" in client.messages.last["system"].lower()
    assert "0000320193" in out                                 # returns the model's text
