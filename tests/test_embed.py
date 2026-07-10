"""embed.py: chunking, pooling, and the cosine delta — no real model needed."""

import numpy as np

from eqd.delta.embed import chunk_text, embed_delta, section_vector


class FakeModel:
    """Deterministic stand-in: same text -> same vector, different -> different."""

    def embed(self, chunks):
        for c in chunks:
            yield np.array([len(c) % 97 + 1.0, c.count("a") + 1.0, c.count("risk") + 1.0])


def test_chunk_text_respects_max_and_keeps_content():
    text = ("We face material risks. " * 300).strip()
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks) == " ".join(text.split())


def test_chunk_text_edge_cases():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []
    assert chunk_text("Short sentence.") == ["Short sentence."]
    # a pathological run-on with no sentence ends still gets split
    assert all(len(c) <= 50 for c in chunk_text("x" * 300, max_chars=50))


def test_section_vector_is_normalized():
    v = section_vector("We face supply risk. Litigation may arise. Demand may fall.", FakeModel())
    assert v is not None
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_section_vector_empty_is_none():
    assert section_vector("", FakeModel()) is None


def test_embed_delta_zero_for_identical_text():
    m = FakeModel()
    text = "Competition may reduce margins. New tariffs were announced this year."
    assert embed_delta(section_vector(text, m), section_vector(text, m)) < 1e-12


def test_embed_delta_positive_for_different_text():
    m = FakeModel()
    a = section_vector("Competition may reduce margins in our core segment.", m)
    b = section_vector("aaaa risk risk risk risk risk aaaa", m)
    assert embed_delta(a, b) > 0
