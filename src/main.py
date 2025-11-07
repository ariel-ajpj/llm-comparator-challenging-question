"""Main entry point for the LLM Comparator."""
import asyncio
import os
from typing import Dict, Optional

from dotenv import load_dotenv

try:
    from .providers.gemini_impl import GeminiProvider
    from .providers.groq_impl import GroqProvider
    from .providers.openai_impl import OpenAIProvider
    from .question_generator import QuestionGenerator
except Exception:
    # Fallback when running the file directly (python src/main.py)
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.providers.gemini_impl import GeminiProvider
    from src.providers.groq_impl import GroqProvider
    from src.providers.openai_impl import OpenAIProvider
    from src.question_generator import QuestionGenerator

from src.models.question import Question
from src.models.response import Response
from src.providers.base import LLMProvider


def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


async def gather_provider_responses(
    providers: Dict[str, LLMProvider],
    question: Question,
    *,
    timeout: Optional[float] = None,
) -> Dict[str, Optional[Response]]:
    responses: Dict[str, Optional[Response]] = {}
    for provider in providers.values():
        try:
            response = await provider.generate_answer(question, timeout=timeout)
            responses[provider.name] = response
            print(f"\n{provider.name.title()} Response:")
            print("-" * 40)
            print(response.answer)
            print("-" * 40)
        except Exception as exc:
            print(f"Error from provider '{provider.name}': {exc}")
            responses[provider.name] = None
    return responses


async def main() -> None:
    """Main entry point."""
    print("LLM Comparator starting...")

    load_dotenv()

    openai_api_key = get_env_var("OPENAI_API_KEY")
    openai_model = "gpt-4"

    groq_api_key = get_env_var("GROQ_API_KEY")
    groq_model = "openai/gpt-oss-120b"

    google_api_key = get_env_var("GOOGLE_API_KEY")
    gemini_model = "gemini-2.5-flash"

    try:
        # Step 1: Generate the question using QuestionGenerator
        generator = QuestionGenerator(api_key=openai_api_key, model=openai_model)
        java_prompt = (
            "Generate a technical interview question about Java that a senior software engineer should be able to answer. "
            "Please search for common questions asked in senior Java interviews and create a similar one."
        )
        print("Generating a senior Java engineering question...")
        question = await generator.generate_question(prompt=java_prompt)
        print("\nGenerated Question:")
        print("-" * 40)
        print(question.text)
        print("-" * 40)
        print(f"\nQuestion ID: {question.id}")

        # Step 2: Send question to multiple providers
        # TODO: Add AnthropicProvider, MockProvider, etc. as needed
        providers: Dict[str, LLMProvider] = {
            "openai": OpenAIProvider(name="openai", model=openai_model, api_key=openai_api_key),
            "groq": GroqProvider(name="groq", model=groq_model, api_key=groq_api_key),
            "gemini": GeminiProvider(name="gemini", model=gemini_model, api_key=google_api_key),
        }

        short_answer_question = Question(
            text=f"{question.text}\n\nPlease answer in no more than three sentences.",
            id=question.id,
            created_at=question.created_at,
        )

        print("\nCollecting responses from providers (short answers requested)...")
        await gather_provider_responses(providers, short_answer_question, timeout=30)
        # TODO: Evaluate and compare responses
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
