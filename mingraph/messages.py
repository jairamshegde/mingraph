from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar


@dataclass(frozen=True)
class ToolCall:
    """The model's request to run one tool: which tool, with what arguments, under which id.
    Arguments are copied and kept read-only, so the request can't change after it's made.
    """
    name: str
    args: Mapping[str, object]
    id: str

    def __post_init__(self):
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))


@dataclass(frozen=True)
class SystemMessage:
    """Instructions for how the model should behave."""
    role: ClassVar[str] = "system"
    content: str

    def __post_init__(self):
        _check_content(self)


@dataclass(frozen=True)
class UserMessage:
    """What the user said."""
    role: ClassVar[str] = "user"
    content: str

    def __post_init__(self):
        _check_content(self)


@dataclass(frozen=True)
class AssistantMessage:
    """The model's reply: text, requests to run tools, or both."""
    role: ClassVar[str] = "assistant"
    content: str
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self):
        _check_content(self)
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        for call in self.tool_calls:
            if not isinstance(call, ToolCall):
                raise TypeError(f"tool_calls must hold ToolCall, got {type(call).__name__}")


@dataclass(frozen=True)
class ToolMessage:
    """A tool's result, answering the tool call with the matching id."""
    role: ClassVar[str] = "tool"
    content: str
    tool_call_id: str

    def __post_init__(self):
        _check_content(self)


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


def _check_content(message) -> None:
    if not isinstance(message.content, str):
        raise TypeError(f"content must be str, got {type(message.content).__name__}")
