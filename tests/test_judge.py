import asyncio
import json

from src.main import judge_responses_with_openai
from src.models.question import Question
from src.models.response import Response


class _FakeHelper:
    """Fake OpenAIClientHelper used to test judge_responses_with_openai."""

    def __init__(self, *_, **__):  # signature-compatible
        self.called_with = []

    async def ask(self, messages, *, model, timeout=None, **kwargs):  # noqa: D401
        """Return a fixed JSON ranking regardless of input."""
        # record a minimal subset of the call for assertions if needed
        self.called_with.append({"model": model, "timeout": timeout, "messages": messages})
        # Rank competitor 2 first, then 1, then 3
        return json.dumps({"results": ["2", "1", "3"]})


async def _run_judge(monkeypatch, responses):
    # Patch OpenAIClientHelper inside src.main to use our fake helper
    import src.main as main_mod

    monkeypatch.setattr(main_mod, "OpenAIClientHelper", _FakeHelper)

    q = Question.create("What is 2+2?")
    await judge_responses_with_openai(
        question=q,
        responses=responses,
        api_key="dummy",
        model="gpt-4",
        timeout=5.0,
    )


def test_judge_basic_ranking(monkeypatch, capsys):
    responses = {
        "openai": Response.create("openai", Question.create("Q"), "A1"),
        "groq": Response.create("groq", Question.create("Q"), "A2"),
        "gemini": Response.create("gemini", Question.create("Q"), "A3"),
    }

    asyncio.run(_run_judge(monkeypatch, responses))

    captured = capsys.readouterr()
    out = captured.out
    # Check that the judge printed a ranking header and mapped competitors back to providers
    assert "Judge ranking (best to worst):" in out
    # Our fake returns ["2", "1", "3"] so groq should be first, openai second, gemini third
    assert "1. groq (competitor 2)" in out
    assert "2. openai (competitor 1)" in out
    assert "3. gemini (competitor 3)" in out


def test_judge_skips_failed_providers(monkeypatch, capsys):
    # Include one failed provider (None) and ensure it's reported as skipped
    responses = {
        "openai": Response.create("openai", Question.create("Q"), "ok"),
        "anthropic": None,
        "groq": Response.create("groq", Question.create("Q"), "ok"),
    }

    asyncio.run(_run_judge(monkeypatch, responses))

    out = capsys.readouterr().out
    assert "will be skipped by the judge" in out
    assert "- anthropic" in out
