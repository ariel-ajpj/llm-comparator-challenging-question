from __future__ import annotations

from typing import Any, List, Mapping, Optional

try:
    # Preferred package-relative imports when running within the package
    from ..models.question import Question
    from ..models.response import Response
    from .base import LLMProvider
    from .openai_client_helper import OpenAIClientHelper
except Exception:  # pragma: no cover - fallback path for direct execution
    # Support direct execution (python src/providers/ollama_impl.py)
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.question import Question
    from src.models.response import Response
    from src.providers.base import LLMProvider
    from src.providers.openai_client_helper import OpenAIClientHelper


class OllamaProvider(LLMProvider):
    """Ollama-backed provider using the OpenAI-compatible HTTP API.

    This reuses OpenAIClientHelper and points it at the local Ollama
    OpenAI-compatible endpoint (default http://localhost:11434/v1).
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        name: str,
        model: str,
        *,
        api_key: str = "ollama-local",  # dummy value; Ollama usually does not require a key
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, model=model)
        if base_url is None:
            base_url = self.DEFAULT_BASE_URL
        # helper is not a dataclass field on the frozen base; set with object.__setattr__
        object.__setattr__(
            self,
            "helper",
            OpenAIClientHelper(api_key=api_key, base_url=base_url),
        )

    async def generate_answer(
        self,
        question: Question,
        *,
        timeout: Optional[float] = None,
        return_raw: bool = False,
        **kwargs: Any,
    ) -> Response:
        """Generate an answer for the given Question using Ollama.

        Assumes Ollama's OpenAI-compatible `/v1/chat/completions` endpoint
        is exposed on the configured base_url.
        """

        messages: List[Mapping[str, str]] = [
            {"role": "user", "content": question.text}
        ]

        result = await self.helper.ask(
            messages,
            model=self.model,
            timeout=timeout,
            return_raw=return_raw,
            **kwargs,
        )

        if result is None:
            raise RuntimeError(
                f"No response from Ollama provider '{self.name}' (model={self.model})"
            )

        if return_raw:
            # extract content from raw ChatCompletion-like object
            try:
                content = result.choices[0].message.content  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError("Unexpected raw response shape from Ollama") from exc
            if not content:
                raise RuntimeError("Empty content in raw Ollama response")
            answer = content.strip()
        else:
            answer = result if isinstance(result, str) else str(result)

        return Response(provider=self.name, question=question, answer=answer)
