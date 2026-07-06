"""Tests for the grounded narrator — mocked, no API call.

The key property to guard: the narrator is only ever handed whitelisted,
pre-computed result fields. It cannot be fed arbitrary/future data, and the
system prompt forbids advice.
"""

from eqd.narrate.memo import event_memo

# --- Minimal OpenAI-shaped mock (chat.completions.create -> choices[0].message.content) ---


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
        self.text = text
        self.last = None

    def create(self, **kwargs):
        self.last = kwargs
        return _Resp(self.text)


class _Chat:
    def __init__(self, text):
        self.completions = _Completions(text)


class _Client:
    def __init__(self, text):
        self.chat = _Chat(text)


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

    msgs = client.chat.completions.last["messages"]
    system = msgs[0]["content"]
    payload = msgs[1]["content"]
    assert "net_added" in payload and "accession" in payload   # grounded facts included
    assert "leaked_future_return" not in payload               # non-whitelisted excluded
    assert "no" in system.lower() and "advice" in system.lower()  # the no-advice rule is stated
    assert "0000320193" in out                                 # returns the model's text


def test_event_memo_strips_reasoning_trace():
    client = _Client("<think>let me reason</think>Final memo (acc 123).")
    out = event_memo({"accession": "123"}, client=client)
    assert out == "Final memo (acc 123)."                      # <think> block removed
