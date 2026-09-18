from dataclasses import dataclass
from typing import Literal, get_args

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One role-tagged turn in a conversation.
    Validated on creation and sealed afterwards, so it can be shared safely.
    """
    role: Role
    content: str

    def __post_init__(self):
        if self.role not in get_args(Role):
            raise ValueError(f"role must be one of {get_args(Role)}, got {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError(f"content must be str, got {type(self.content).__name__}")
