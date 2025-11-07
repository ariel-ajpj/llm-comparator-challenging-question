import asyncio

import pytest

from src.models.question import Question
from src.models.response import Response
from src.providers.base import LLMProvider


class DummyProvider(LLMProvider):
    async def generate_answer(self, question: Question) -> Response:
        return Response(provider=self.name, question=question, answer="ok")


def test_llmprovider_validation():
    with pytest.raises(ValueError):
        DummyProvider(name="", model="m")
    with pytest.raises(ValueError):
        DummyProvider(name="p", model="  ")


def test_repr_and_generate_answer():
    p = DummyProvider(name="d", model="m")
    q = Question.create("hello")
    res = asyncio.run(p.generate_answer(q))
    assert isinstance(res, Response)
    assert res.answer == "ok"
    assert "d" in repr(p)
