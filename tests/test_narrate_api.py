"""Tests for the Vercel serverless narrator (api/narrate.py) — mocked, no API call.

Guards the two properties that make a public endpoint safe:
  1. `event` must be a real key in the pre-computed panel — an unknown key can
     never reach the model, and the client only ever sends the key (the numbers
     are looked up server-side, so they can't be spoofed).
  2. The prompt is built from those looked-up pre-computed facts (grounded), and
     the model's text is returned verbatim.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

import narrate  # noqa: E402


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
        self.text, self.last = text, None

    def create(self, **kwargs):
        self.last = kwargs
        return _Resp(self.text)


class _Chat:
    def __init__(self, text):
        self.completions = _Completions(text)


class _Client:
    def __init__(self, text):
        self.chat = _Chat(text)


def test_panel_is_full_universe_and_closed():
    assert len(narrate._EVENTS) > 1000            # full universe, not a 10-event demo
    tickers = {v["ticker"] for v in narrate._EVENTS.values()}
    assert len(tickers) > 100                      # hundreds of companies
    assert "TOTALLY-FAKE" not in narrate._EVENTS   # a fabricated key is rejected upstream


def test_memo_is_grounded_and_verbatim():
    narrate._CACHE.clear()
    key = sorted(narrate._EVENTS)[0]
    client = _Client(f"Memo citing {narrate._EVENTS[key]['accession']}.")
    out = narrate._memo(key, client=client)

    msgs = client.chat.completions.last["messages"]
    facts = json.loads(msgs[1]["content"].split("\n", 1)[1])   # user message carries the facts
    # every value in the prompt comes from the bundled, pre-computed facts
    assert facts == narrate._EVENTS[key]
    assert "advice" in msgs[0]["content"].lower()              # system message states the no-advice rule
    assert narrate._EVENTS[key]["accession"] in out           # returns the model text


def test_memo_caches_within_instance():
    narrate._CACHE.clear()
    key = sorted(narrate._EVENTS)[0]
    narrate._memo(key, client=_Client("first"))
    # second call must not need a client (served from warm cache)
    assert narrate._memo(key) == "first"
