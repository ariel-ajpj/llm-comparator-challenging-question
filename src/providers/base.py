
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.models.question import Question
from src.models.response import Response


# Keeping this protocol as a future option if we need to:
# - Support third-party LLM implementations without inheritance
# - Add runtime type checking for provider-like objects
# - Create test mocks without full ABC implementation
@runtime_checkable
class LLMProviderProtocol(Protocol):
    name: str
    model: str

    async def generate_answer(self, question: Question) -> Response:
        ...


@dataclass(frozen=True)
class LLMProvider(ABC):
    name: str = field()
    model: str = field()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Provider name cannot be empty")
        if not self.model.strip():
            raise ValueError("Model identifier cannot be empty")

    @abstractmethod
    async def generate_answer(self, question: Question) -> Response:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"

