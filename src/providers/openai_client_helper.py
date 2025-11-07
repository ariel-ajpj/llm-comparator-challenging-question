from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Union

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class OpenAIClientHelper:

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        organization: str | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("OpenAIClientHelper: `api_key` must be a non-empty string.")
        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url, organization=organization)

    async def ask(
        self,
        messages: List[ChatCompletionMessageParam],
        *,
        model: str,
        return_raw: bool = False,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Union[str, ChatCompletion]]:
        """
        Call Chat Completions and return the first message content (default),
        or the raw ChatCompletion object if `return_raw=True`.

        Args:
            messages: List of chat messages (role/content dicts)
            model: Model name to use
            return_raw: If True, return the full ChatCompletion response
            timeout: Optional timeout in seconds for the request
            **kwargs: Forwarded to the OpenAI client (temperature, max_tokens, etc.)

        Returns:
            The first message content (str), the raw ChatCompletion, or None on error.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAIClientHelper.ask: `model` must be a non-empty string.")
        model = model.strip()

        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("OpenAIClientHelper.ask: `messages` must be a non-empty list.")

        def _sync_call() -> ChatCompletion:
            return self.client.chat.completions.create(model=model, messages=messages, **kwargs)

        try:
            if timeout is not None:
                response: ChatCompletion = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=timeout)
            else:
                response = await asyncio.to_thread(_sync_call)
        except asyncio.TimeoutError:
            logger.exception("OpenAI request timed out for model %s", model)
            return None
        except Exception as e:
            # Catch any SDK or runtime error. We avoid importing SDK-specific
            # error types directly to be resilient to different openai package
            # versions installed in the environment.
            logger.exception("OpenAI SDK error calling model %s: %s", model, e)
            return None

        if return_raw:
            return response

        if not response or not getattr(response, "choices", None):
            logger.warning("No choices returned in OpenAI response for model %s.", model)
            return None

        msg = response.choices[0].message
        if not msg or not getattr(msg, "content", None):
            logger.warning("Response message has no content.")
            return None

        return msg.content.strip()
