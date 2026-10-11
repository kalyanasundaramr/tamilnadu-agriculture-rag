"""Lightweight tests for retrieval behavior. Run: pytest -q"""
from retriever import INSUFFICIENT

def test_insufficient_answer_is_explicit():
    assert "could not find sufficient information" in INSUFFICIENT.lower()
    assert "official agriculture department" in INSUFFICIENT.lower()
