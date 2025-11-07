from __future__ import annotations

from typing import Any, List, Mapping, Optional

from src.models.question import Question
from src.models.response import Response

from .base import LLMProvider
from .openai_client_helper import OpenAIClientHelper


class OpenAIProvider(LLMProvider):
    """OpenAI-backed provider using OpenAIClientHelper."""

    def __init__(
        self,
        name: str,
        model: str,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, model=model)
        # helper is not a dataclass field on the frozen base; set with object.__setattr__
        object.__setattr__(self, "helper", OpenAIClientHelper(api_key=api_key, base_url=base_url, organization=organization))

    async def generate_answer(self, question: Question, *, timeout: Optional[float] = None, return_raw: bool = False, **kwargs: Any) -> Response:
        """Generate an answer for the given Question using OpenAI."""
        messages: List[Mapping[str, str]] = [{"role": "user", "content": question.text}]

        result = await self.helper.ask(messages, model=self.model, timeout=timeout, return_raw=return_raw, **kwargs)

        if result is None:
            raise RuntimeError(f"No response from OpenAI provider '{self.name}' (model={self.model})")

        if return_raw:
            # extract content from raw ChatCompletion
            try:
                content = result.choices[0].message.content  # type: ignore[attr-defined]
            except Exception:
                raise RuntimeError("Unexpected raw response shape from OpenAI")
            if not content:
                raise RuntimeError("Empty content in raw OpenAI response")
            answer = content.strip()
        else:
            answer = result if isinstance(result, str) else str(result)

        return Response(provider=self.name, question=question, answer=answer)
