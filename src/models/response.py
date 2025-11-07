
from dataclasses import dataclass, field
from typing import Self

from src.models.question import Question


@dataclass(slots=True, frozen=True)
class Response:
    """
    Represents a response from an LLM provider.
    """
    provider: str = field(
        metadata={"help": "Name of the LLM provider"}
    )
    question: Question = field(
        metadata={"help": "The question this response is for"}
    )
    answer: str = field(
        metadata={"help": "The answer provided by the LLM"}
    )

    def __post_init__(self) -> None:
        """Validate the response attributes after initialization."""
        if not self.provider.strip():
            raise ValueError("Provider name cannot be empty")
        if not self.answer.strip():
            raise ValueError("Answer text cannot be empty")
        if not self.question:
            raise ValueError("Response must be associated with a valid Question")

    @classmethod
    def create(cls, provider: str, question: Question, answer: str) -> Self:
        return cls(provider=provider, question=question, answer=answer)

    def short_preview(self, length: int = 80) -> str:
        """
        Get a truncated version of the answer text.
        """
        return self.answer if len(self.answer) <= length else f"{self.answer[:length]}..."

