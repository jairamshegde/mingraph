from abc import ABC, abstractmethod

from mingraph.messages import Message, SystemMessage, UserMessage

Turn = list[Message]


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
