from __future__ import annotations

from typing import Any, List, Mapping, Optional

try:
    # Preferred package-relative imports when used as part of the package
    from ..models.question import Question
    from ..models.response import Response
    from .base import LLMProvider
    from .openai_client_helper import OpenAIClientHelper
except Exception:
    # Fallback for direct script execution (python src/providers/groq_impl.py)
    import sys
    from pathlib import Path

    # project_root is two levels up from src/providers -> project root
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.question import Question
    from src.models.response import Response
    from src.providers.base import LLMProvider
    from src.providers.openai_client_helper import OpenAIClientHelper


class GroqProvider(LLMProvider):
    """Groq-backed provider using the same OpenAI-compatible client helper.

    Groq exposes an OpenAI-compatible API; we reuse OpenAIClientHelper but
    default the base_url to Groq's endpoint. The rest of the behaviour mirrors
    the OpenAIProvider implementation.
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

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
        if base_url is None:
            base_url = self.DEFAULT_BASE_URL
        # helper is not a dataclass field on the frozen base; set with object.__setattr__
        object.__setattr__(self, "helper", OpenAIClientHelper(api_key=api_key, base_url=base_url, organization=organization))

    async def generate_answer(self, question: Question, *, timeout: Optional[float] = None, return_raw: bool = False, **kwargs: Any) -> Response:
        """Generate an answer for the given Question using Groq (OpenAI-compatible API)."""
        messages: List[Mapping[str, str]] = [{"role": "user", "content": question.text}]

        result = await self.helper.ask(messages, model=self.model, timeout=timeout, return_raw=return_raw, **kwargs)

        if result is None:
            raise RuntimeError(f"No response from Groq provider '{self.name}' (model={self.model})")

        if return_raw:
            # extract content from raw ChatCompletion
            try:
                content = result.choices[0].message.content  # type: ignore[attr-defined]
            except Exception:
                raise RuntimeError("Unexpected raw response shape from Groq/OpenAI-compatible API")
            if not content:
                raise RuntimeError("Empty content in raw Groq response")
            answer = content.strip()
        else:
            answer = result if isinstance(result, str) else str(result)

        return Response(provider=self.name, question=question, answer=answer)
