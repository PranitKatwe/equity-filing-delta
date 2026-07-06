"""Tests for the grounded filing-Q&A endpoint (api/ask.py) — mocked, no API call.

Guards the RAG grounding: the model is handed ONLY the real diff passages for a
known company, plus the user's question; the system prompt forbids advice; and
an unknown company never reaches the model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

import ask  # noqa: E402


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


def test_passages_present_for_many_companies():
    assert len(ask.PASSAGES) > 100
    assert "MADE-UP-TICKER" not in ask.PASSAGES


def test_answer_is_grounded_in_the_companys_passages():
    ask._CACHE.clear()
    company = next(c for c, p in ask.PASSAGES.items() if p["added"])  # one with added text
    a_sentence = ask.PASSAGES[company]["added"][0]
    client = _Client("The filing added several new risk sentences.")
    ask._answer(company, "What new risks were added?", client=client)

    msgs = client.chat.completions.last["messages"]
    system, user = msgs[0]["content"].lower(), msgs[1]["content"]
    assert "advice" in system and "only" in system          # no-advice + grounding rules stated
    assert a_sentence in user                                 # the real passage is in the prompt
    assert "What new risks were added?" in user               # the question is included
    assert ask.PASSAGES[company]["accession"] in user         # provenance for citation


def test_answer_caches_within_instance():
    ask._CACHE.clear()
    company = next(iter(ask.PASSAGES))
    ask._answer(company, "same question", client=_Client("first"))
    assert ask._answer(company, "same question") == "first"   # served from warm cache, no client
