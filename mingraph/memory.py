from abc import ABC, abstractmethod

from mingraph.llm import BaseLLM
from mingraph.messages import AssistantMessage, Message, SystemMessage, UserMessage

Turn = list[Message]

_RECAP_PREFIX = "Here is a summary of the conversation to date:\n\n"
_SUMMARISE_INSTRUCTION = (
    "Summarise the conversation below. Keep every fact, name, date and decision "
    "that may matter later. Reply with the summary only.\n\n"
)


class Memory(ABC):
    """Keeps the whole conversation and decides what the model sees each call.
    The record only grows; strategies choose which turns to send from it.
    """
    def __init__(self, system: SystemMessage):
        self._system = system
        self._chat: list[Message] = []

    def add(self, message: Message) -> None:
        """Record something that happened: a user message, a reply, or a tool result."""
        if isinstance(message, SystemMessage):
            raise TypeError("the system message is given once, when the memory is created")
        self._chat.append(message)

    def messages(self) -> list[Message]:
        """Build the list to send for this call: the system message, then the strategy's pick."""
        return [self._system, *self._pick(self._turns())]

    def _turns(self) -> list[Turn]:
        # A turn is a user message plus everything after it, up to the next user message.
        turns: list[Turn] = []
        for message in self._chat:
            if isinstance(message, UserMessage) or not turns:
                turns.append([])
            turns[-1].append(message)
        return turns

    @abstractmethod
    def _pick(self, turns: list[Turn]) -> list[Message]:
        """Choose what to send from the turns, as a flat list of messages in order."""
        raise NotImplementedError


class KeepAll(Memory):
    """Send every turn."""
    def _pick(self, turns: list[Turn]) -> list[Message]:
        return [m for turn in turns for m in turn]


class LastN(Memory):
    """Send only the last n turns."""
    def __init__(self, n: int, system: SystemMessage):
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        super().__init__(system)
        self._n = n

    def _pick(self, turns: list[Turn]) -> list[Message]:
        return [m for turn in turns[-self._n:] for m in turn]


class Summarise(Memory):
    """Send a recap of older turns, plus every turn the recap doesn't cover yet.
    The recap is rewritten only when more than `trigger` turns are uncovered,
    and then covers all but the last `keep` turns.
    """
    def __init__(self, llm: BaseLLM, keep: int, trigger: int, system: SystemMessage):
        if not 1 <= keep < trigger:
            raise ValueError(f"need 1 <= keep < trigger, got keep={keep}, trigger={trigger}")
        super().__init__(system)
        self._llm = llm
        self._keep = keep
        self._trigger = trigger
        self._recap: str | None = None
        self._covered = 0  # how many turns, from the start, the recap covers

    def _pick(self, turns: list[Turn]) -> list[Message]:
        uncovered = turns[self._covered:]
        if len(uncovered) > self._trigger:
            to_cover = uncovered[:-self._keep]
            self._recap = self._rewrite(to_cover)
            self._covered += len(to_cover)
            uncovered = uncovered[-self._keep:]
        picked = [m for turn in uncovered for m in turn]
        if self._recap is None:
            return picked
        return [UserMessage(_RECAP_PREFIX + self._recap), *picked]

    def _rewrite(self, turns: list[Turn]) -> str:
        # Rolling: the old recap plus the newly covered turns, as a plain-text printout.
        lines = [f"Summary so far: {self._recap}"] if self._recap else []
        lines += [_printout(m) for turn in turns for m in turn]
        response = self._llm.generate([UserMessage(_SUMMARISE_INSTRUCTION + "\n".join(lines))])
        return response.message.content


def _printout(message: Message) -> str:
    parts = [message.content] if message.content else []
    if isinstance(message, AssistantMessage):
        parts += [f"(called {c.name} {dict(c.args)})" for c in message.tool_calls]
    return f"{message.role}: {' '.join(parts)}"
