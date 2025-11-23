"""Main entry point for the LLM Comparator."""
import asyncio
import json
import os
from typing import Dict, Optional

from dotenv import load_dotenv

try:
    from .providers.anthropic_impl import AnthropicProvider
    from .providers.gemini_impl import GeminiProvider
    from .providers.groq_impl import GroqProvider
    from .providers.ollama_impl import OllamaProvider
    from .providers.openai_client_helper import OpenAIClientHelper
    from .providers.openai_impl import OpenAIProvider
    from .question_generator import QuestionGenerator
except Exception:
    # Fallback when running the file directly (python src/main.py)
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.providers.anthropic_impl import AnthropicProvider
    from src.providers.gemini_impl import GeminiProvider
    from src.providers.groq_impl import GroqProvider
    from src.providers.ollama_impl import OllamaProvider
    from src.providers.openai_client_helper import OpenAIClientHelper
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


async def judge_responses_with_openai(
    *,
    question: Question,
    responses: Dict[str, Optional[Response]],
    api_key: str,
    model: str,
    timeout: Optional[float] = 30.0,
) -> None:
    """Use an OpenAI model to rank provider responses without revealing identities.

    The judge sees anonymous competitors (1..N) and returns a JSON ranking
    like {"results": ["1", "3", "2"]}. We then map that back to
    provider names for display.
    """

    # Filter out missing responses
    failed_providers = [name for name, resp in responses.items() if resp is None]
    if failed_providers:
        print("\nThe following providers had no valid response and will be skipped by the judge:")
        for name in failed_providers:
            print(f"  - {name}")

    competitors: list[tuple[str, Response]] = [
        (name, resp)
        for name, resp in responses.items()
        if resp is not None
    ]

    if len(competitors) < 2:
        print("\nNot enough valid responses to perform judging.")
        return

    # Build anonymized responses text
    numbered_to_provider: Dict[int, str] = {}
    together_parts: list[str] = []
    for idx, (provider_name, resp) in enumerate(competitors, start=1):
        numbered_to_provider[idx] = provider_name
        together_parts.append(
            f"Competitor {idx}:\n{resp.answer}\n"
        )
    together = "\n".join(together_parts).strip()

    question_text = question.text

    judge_prompt = (
        f"You are judging a competition between {len(competitors)} competitors.\n"
        "Each model has been given this question:\n\n"
        f"{question_text}\n\n"
        "Your job is to evaluate each response for clarity and strength of argument, "
        "and rank them in order of best to worst.\n"
        "Respond with JSON, and only JSON, with the following format:\n"
        '{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}\n\n'
        "Here are the responses from each competitor:\n\n"
        f"{together}\n\n"
        "Now respond with the JSON with the ranked order of the competitors, nothing else. "
        "Do not include markdown formatting or code blocks."
    )

    helper = OpenAIClientHelper(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are an impartial judge of answer quality."},
        {"role": "user", "content": judge_prompt},
    ]

    print("\nAsking OpenAI judge to rank the responses...")
    raw = await helper.ask(messages, model=model, timeout=timeout)
    if not raw:
        print("Judge did not return a response.")
        return

    try:
        data = json.loads(raw)
        order = data.get("results")
        if not isinstance(order, list):
            raise ValueError("`results` must be a list")
    except Exception as exc:
        print("Failed to parse judge JSON response:", exc)
        print("Raw judge output:", raw)
        return

    print("\nJudge ranking (best to worst):")
    for rank, item in enumerate(order, start=1):
        try:
            num = int(str(item))
        except ValueError:
            print(f"  Rank {rank}: invalid competitor identifier {item!r}")
            continue
        provider_name = numbered_to_provider.get(num)
        if not provider_name:
            print(f"  Rank {rank}: unknown competitor number {num}")
            continue
        print(f"  {rank}. {provider_name} (competitor {num})")


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

    anthropic_api_key = get_env_var("ANTHROPIC_API_KEY")
    anthropic_model = "claude-3-5-sonnet-20241022"

    ollama_model = "llama3.2:latest"

    try:
        # Step 1: Generate the question using QuestionGenerator
        generator = QuestionGenerator(api_key=openai_api_key, model=openai_model)
        # java_prompt = (
        #     "Generate a technical interview question about Java that a senior software engineer should be able to answer. "
        #     "Please search for common questions asked in senior Java interviews and create a similar one."
        # )
        # print("Generating a senior Java engineering question...")
        # question = await generator.generate_question(prompt=java_prompt)
        print("Generating a hard question...")
        question = await generator.generate_question()
        print("\nGenerated Question:")
        print("-" * 40)
        print(question.text)
        print("-" * 40)
        print(f"\nQuestion ID: {question.id}")

        # Step 2: Send question to multiple providers
        providers: Dict[str, LLMProvider] = {
            "openai": OpenAIProvider(name="openai", model=openai_model, api_key=openai_api_key),
            "groq": GroqProvider(name="groq", model=groq_model, api_key=groq_api_key),
            "gemini": GeminiProvider(name="gemini", model=gemini_model, api_key=google_api_key),
            "anthropic": AnthropicProvider(name="anthropic", model=anthropic_model, api_key=anthropic_api_key),
            "ollama": OllamaProvider(name="ollama", model=ollama_model),
        }

        short_answer_question = Question(
            text=f"{question.text}\n\nPlease answer in no more than three sentences.",
            id=question.id,
            created_at=question.created_at,
        )

        print("\nCollecting responses from providers (short answers requested)...")
        provider_responses = await gather_provider_responses(providers, short_answer_question, timeout=30)

        await judge_responses_with_openai(
            question=short_answer_question,
            responses=provider_responses,
            api_key=openai_api_key,
            model=openai_model,
            timeout=30.0,
        )
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
