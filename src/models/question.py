from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Self


@dataclass(slots=True, frozen=True)
class Question:
    """
    Represents a question to be processed by multiple LLM providers.
    """
    text: str = field(
        metadata={"help": "The actual question text"}
    )
    id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"help": "Unique identifier for the question"}
    )
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
        metadata={"help": "Timestamp when the question was created"}
    )

    def __post_init__(self) -> None:
        """Validate the question attributes after initialization."""
        if not self.text.strip():
            raise ValueError("Question text cannot be empty")

    @classmethod
    def create(cls, text: str) -> Self:
        return cls(text=text)

    def short_preview(self, length: int = 80) -> str:
        """
        Get a truncated version of the question text.
        """
        return self.text if len(self.text) <= length else f"{self.text[:length]}..."

