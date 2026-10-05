from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mingraph.messages import AssistantMessage, Message
from mingraph.tools import Tool

StopReason = Literal["stop", "length", "tool_call", "other"]


@dataclass(frozen=True)
class LLMResponse:
    """The assistant's reply plus metadata about the call that produced it.
    Token counts are None when the provider didn't report them; output_tokens includes any thinking.
    The thinking text and its token count are None when the provider doesn't return them.
    """
    message: AssistantMessage
    input_tokens: int | None
    output_tokens: int | None
    stop_reason: StopReason
    reasoning: str | None = None
    reasoning_tokens: int | None = None


class BaseLLM(ABC):
    """Contract every LLM provider must satisfy.
    Callers depend only on this interface, never on a specific provider.
    """
    @abstractmethod
    def generate(self, messages: list[Message], tools: Sequence[Tool] = ()) -> LLMResponse:
        """Send the conversation, plus any tools on offer, to the model and return its reply."""
        raise NotImplementedError
