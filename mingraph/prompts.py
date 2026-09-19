from dataclasses import dataclass, replace
from string import Formatter

from mingraph.messages import Message


@dataclass(frozen=True)
class MessagesPlaceholder:
    """A slot in a template where a list of messages, e.g. history, is inserted."""
    name: str


class ChatPromptTemplate:
    """Builds a list of messages from template messages with {variables} and placeholder slots.
    Variables must be plain names. Filling fails on any missing or unknown variable.
    """

    def __init__(self, parts: list[Message | MessagesPlaceholder]):
        self._parts = tuple(parts)
        variables = set()
        for part in self._parts:
            if isinstance(part, MessagesPlaceholder):
                variables.add(part.name)
            elif isinstance(part, Message):
                variables.update(_variables_in(part.content))
            else:
                raise TypeError(f"parts must be Message or MessagesPlaceholder, got {type(part).__name__}")
        self._variables = frozenset(variables)

    @property
    def variables(self) -> frozenset[str]:
        return self._variables

    def format_messages(self, **values) -> list[Message]:
        missing = self._variables - values.keys()
        unknown = values.keys() - self._variables
        if missing or unknown:
            raise ValueError(
                f"template expects {sorted(self._variables)}; "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        messages = []
        for part in self._parts:
            if isinstance(part, MessagesPlaceholder):
                inserted = values[part.name]
                if not isinstance(inserted, list) or not all(isinstance(m, Message) for m in inserted):
                    raise TypeError(f"{part.name!r} must be a list of Message")
                messages.extend(inserted)
            else:
                messages.append(replace(part, content=part.content.format(**values)))
        return messages


def _variables_in(text: str) -> set[str]:
    names = set()
    for _, name, spec, conversion in Formatter().parse(text):
        if name is None:
            continue
        if not name.isidentifier() or spec or conversion:
            raise ValueError(
                f"template variables must be plain names, got {name!r} in {text!r} "
                f"(write literal braces as {{{{ and }}}})"
            )
        names.add(name)
    return names
