import pytest

from src.models.question import Question
from src.models.response import Response


def test_question_create_and_preview():
    q = Question.create("What is 2+2?")
    assert q.text == "What is 2+2?"
    assert isinstance(q.id, str) and len(q.id) > 0
    preview = q.short_preview(5)
    assert isinstance(preview, str)


def test_question_empty_text_raises():
    with pytest.raises(ValueError):
        Question.create("   ")


def test_response_create_and_preview():
    q = Question.create("Explain recursion")
    r = Response.create("openai", q, "Recursion is when a function calls itself.")
    assert r.provider == "openai"
    assert r.question is q
    assert "Recursion" in r.answer
    assert isinstance(r.short_preview(10), str)


def test_response_validation_errors():
    q = Question.create("Q")
    with pytest.raises(ValueError):
        Response.create("", q, "ans")
    with pytest.raises(ValueError):
        Response.create("openai", q, "   ")
