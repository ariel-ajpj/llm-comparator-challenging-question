# LLM Comparator for Challenging Questions

A Python sample project that implements an LLM orchestration and parallelization pattern to compare responses from multiple language models.

## Overview

This project demonstrates how to coordinate multiple Large Language Models (LLMs) to solve complex questions and compare their performance. It implements an asynchronous pattern for parallel execution and evaluation of responses.

## Process Flow

1. **Question Generation**: Use an LLM to generate a challenging question
2. **Parallel Processing**: Send the question to multiple LLMs simultaneously
   - Asynchronous execution
   - Configurable timeout per model
   - Wait for all responses or timeout
3. **Response Collection**: Aggregate all received answers
4. **Blind Evaluation**: Have a separate LLM rank the answers
   - Model identities are hidden to prevent bias
   - Evaluation based on response quality only
5. **Results Presentation**: Synthesize and display results in a clean, formatted output

## Custom Questions

The sample entry point (`src/main.py`) currently seeds the flow with a senior-level Java interview prompt (`java_prompt`). Update that string to target any other domain (e.g., Python systems design, math puzzles) and the generator will craft a question accordingly before dispatching it to the configured providers.

## AI Providers

This example project compares responses from 4 leading AI providers:

1. **OpenAI** - Known for GPT-4 and advanced language models
2. **Anthropic** - Creators of Claude models with strong reasoning
3. **Google** - Offers PaLM and Gemini models
4. **Groq** - Known for fast inference of LLM models

### Environment Setup

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

## TODO (future improvements)

- Add retries and exponential backoff for transient failures (currently no retries/backoff).
- Revisit timeout and request-level configuration (basic timeout support added in code, but could be centralized).
- Consider returning richer response metadata (role, finish_reason) or an option to always return raw responses.
- Add support for a local Ollama provider.
- Create the tests to all providers.
- Allow passing a custom question prompt as a CLI argument or configuration parameter instead of editing `src/main.py`.
