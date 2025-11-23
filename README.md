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

The sample entry point (`src/main.py`) currently asks OpenAI to generate a single **hard question** by default (no explicit domain). You can customize this by editing `main.py` to pass your own prompt into `QuestionGenerator.generate_question(prompt=...)` (for example, to focus on Java, Python systems design, math puzzles, etc.). The generator will craft a question accordingly before dispatching it to the configured providers.

## AI Providers

This example project compares responses from multiple AI providers:

1. **OpenAI** - Known for GPT-4 and advanced language models (also used as the judge)
2. **Anthropic** - Creators of Claude models with strong reasoning
3. **Google** - Offers Gemini models via an OpenAI-compatible endpoint
4. **Groq** - Known for fast inference of LLM models via an OpenAI-compatible endpoint
5. **Ollama (local)** - Runs local models (e.g., `llama3.2:latest`) via an OpenAI-compatible HTTP API on `http://localhost:11434/v1`

### Environment Setup

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Ollama is assumed to be running locally on `http://localhost:11434` and does **not** require an API key; the code uses an OpenAI-compatible `/v1/chat/completions` endpoint.

## TODO (future improvements)

- Add retries and exponential backoff for transient failures (currently no retries/backoff).
- Revisit timeout and request-level configuration (basic timeout support added in code, but could be centralized).
- Consider returning richer response metadata (role, finish_reason) or an option to always return raw responses.
- Improve configurability for local providers (e.g., Ollama base URL, model selection via config/CLI).
- Expand the test suite to cover more providers and the full judge flow.
- Allow passing a custom question prompt as a CLI argument or configuration parameter instead of editing `src/main.py`.
