"""Question generation using OpenAI."""
from typing import Optional

try:
    from .models.question import Question
    from .providers.openai_client_helper import OpenAIClientHelper
except Exception:
    # Fallback when running the file directly (python src/question_generator.py)
    # Ensure project root is importable and import package-style so
    # relative imports inside submodules resolve correctly.
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.question import Question
    from src.providers.openai_client_helper import OpenAIClientHelper


class QuestionGenerator:
    """Generates challenging questions using OpenAI."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[float] = 30.0,
    ) -> None:
        self._helper = OpenAIClientHelper(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self.model = model
        self.timeout = timeout

    async def generate_question(self, prompt: str = None) -> Question:
        """Generate a challenging question using OpenAI. Accepts an optional custom prompt."""
        if prompt is None:
            # Default system prompt for challenging questions
            system_content = (
                "You are an expert at creating challenging but clear questions. "
                "Generate ONE challenging question that:\n"
                "1. Tests reasoning and analytical capabilities\n"
                "2. Has no obvious answer but can be reasoned about\n"
                "3. Requires detailed explanation and analysis\n"
                "4. Is clear and unambiguous\n"
                "5. Can be answered without external resources\n\n"
                "Return ONLY the question text, no preamble or explanation."
            )
            user_content = "Generate a challenging question."
        else:
            # Use custom prompt for both system and user message
            system_content = "Please keep the answer short"
            user_content = prompt

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        response = await self._helper.ask(
            messages=messages,
            model=self.model,
            timeout=self.timeout,
            temperature=1.0,
            max_tokens=200,
        )

        if not response:
            raise RuntimeError("Failed to generate question using OpenAI")

        question_text = response.strip()
        return Question.create(text=question_text)
