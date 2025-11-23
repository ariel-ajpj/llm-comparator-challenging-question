from __future__ import annotations

import asyncio
from typing import Any, List, Mapping, Optional

try:
    # Preferred package-relative imports when running within the package
    from ..models.question import Question
    from ..models.response import Response
    from .base import LLMProvider
except Exception:  # pragma: no cover - fallback path for direct execution
    # Support direct execution (python src/providers/anthropic_impl.py)
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.question import Question
    from src.models.response import Response
    from src.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic provider using the official `anthropic` SDK.

    This implementation calls the Messages API and wraps the response
    into the common Response model used across providers.
    """

    def __init__(
        self,
        name: str,
        model: str,
        *,
        api_key: str,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, model=model)

        # Import lazily so tests can stub the anthropic package if desired
        from anthropic import Anthropic

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        # Store the low-level client on the frozen dataclass instance
        object.__setattr__(self, "_client", Anthropic(**client_kwargs))

    async def generate_answer(
        self,
        question: Question,
        *,
        timeout: Optional[float] = None,
        return_raw: bool = False,
        **kwargs: Any,
    ) -> Response:
        """Generate an answer for the given Question using Anthropic.

        The interface mirrors the other providers: it returns a Response
        object with the provider name, original Question instance, and
        text answer. If `return_raw` is True, the raw SDK response object
        is exposed via the Response.answer field as a stringified object.
        """

        # Anthropic's Messages API expects a list of user messages.
        messages: List[Mapping[str, str]] = [
            {"role": "user", "content": question.text}
        ]

        def _call_sync():
            # Import inside to avoid hard dependency at module import time

            return self._client.messages.create(  # type: ignore[attr-defined]
                model=self.model,
                max_tokens=512,
                messages=messages,
                **kwargs,
            )

        if timeout is not None:
            resp = await asyncio.wait_for(asyncio.to_thread(_call_sync), timeout=timeout)
        else:
            resp = await asyncio.to_thread(_call_sync)

        if resp is None:
            raise RuntimeError(f"No response from Anthropic provider '{self.name}' (model={self.model})")

        if return_raw:
            answer_text = str(resp)
        else:
            # Extract the first text block from the response content
            content = getattr(resp, "content", None)
            if not content:
                raise RuntimeError("Empty content in Anthropic response")

            # Messages API typically returns a list of content blocks; we
            # join all text segments for simplicity.
            parts: List[str] = []
            for block in content:
                block_text = getattr(block, "text", None) or getattr(block, "content", None)
                if isinstance(block_text, str):
                    parts.append(block_text)
            if not parts:
                raise RuntimeError("Could not extract text from Anthropic response content")

            answer_text = "\n".join(parts).strip()

        return Response(provider=self.name, question=question, answer=answer_text)
