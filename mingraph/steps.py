import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence as Seq

from mingraph.llm import BaseLLM
from mingraph.prompts import ChatPromptTemplate
from mingraph.retrievers import Retriever
from mingraph.tools import Tool, ToolRegistry

State = dict[str, object]


class Step(ABC):
    """One unit of work over a shared state.
    A container of steps is itself a step, so it fits anywhere a single step does.
    """
    @abstractmethod
    def run(self, state: State) -> State:
        """Read what's needed from the state; return only the slots this step set."""
        raise NotImplementedError


class Sequence(Step):
    """Runs steps in order. Each step sees the state plus every earlier step's update.
    Returns everything its steps added, as one update, like any other step.
    """
    def __init__(self, steps: list[Step]):
        self._steps = list(steps)

    def run(self, state: State) -> State:
        added: State = {}
        for step in self._steps:
            added = {**added, **step.run({**state, **added})}
        return added


class Branch(Step):
    """Runs exactly one of its steps, picked by `choose`, and returns that step's update.
    `choose` reads the state and returns a name from `steps`. It picks; it doesn't cook.
    """
    def __init__(self, choose: Callable[[State], str], steps: dict[str, Step]):
        self._choose = choose
        self._steps = dict(steps)

    def run(self, state: State) -> State:
        name = self._choose(state)
        if name not in self._steps:
            raise KeyError(f"choose returned {name!r}, expected one of {list(self._steps)}")
        return self._steps[name].run(state)


class Parallel(Step):
    """Runs every step on the same state; none sees another's update.
    Returns all their updates as one. Two steps setting the same slot is an error.
    """
    def __init__(self, steps: list[Step]):
        self._steps = list(steps)

    def run(self, state: State) -> State:
        added: State = {}
        for step in self._steps:
            update = step.run(state)
            clash = added.keys() & update.keys()
            if clash:
                raise ValueError(f"more than one step set {sorted(clash)}")
            added = {**added, **update}
        return added


class Prompt(Step):
    """Fills a template from the state slots named after its variables."""
    def __init__(self, template: ChatPromptTemplate, write: str = "messages"):
        self._template = template
        self._write = write

    def run(self, state: State) -> State:
        values = {name: state[name] for name in self._template.variables}
        return {self._write: self._template.format_messages(**values)}


class CallModel(Step):
    """Sends the messages in one slot to the model and puts its reply in another.
    Only the reply goes in the state; the token counts stay on the response.
    """
    def __init__(self, llm: BaseLLM, tools: Seq[Tool] = (), read: str = "messages", write: str = "reply"):
        self._llm = llm
        self._tools = tuple(tools)
        self._read = read
        self._write = write

    def run(self, state: State) -> State:
        return {self._write: self._llm.generate(state[self._read], self._tools).message}


class Text(Step):
    """Puts the text of the reply in one slot into another slot, so a later prompt can use it."""
    def __init__(self, read: str = "reply", write: str = "text"):
        self._read = read
        self._write = write

    def run(self, state: State) -> State:
        return {self._write: state[self._read].content}


class RunTools(Step):
    """Runs every tool call in a reply and puts the answers in a slot, in call order."""
    def __init__(self, registry: ToolRegistry, read: str = "reply", write: str = "tool_results"):
        self._registry = registry
        self._read = read
        self._write = write

    def run(self, state: State) -> State:
        return {self._write: [self._registry.run(call) for call in state[self._read].tool_calls]}


class Retrieve(Step):
    """Asks a retriever for the documents matching the query in one slot and puts them in another."""
    def __init__(self, retriever: Retriever, read: str = "question", write: str = "docs"):
        self._retriever = retriever
        self._read = read
        self._write = write

    def run(self, state: State) -> State:
        return {self._write: self._retriever.invoke(state[self._read])}


class FormatDocs(Step):
    """Joins the text of the documents in one slot, blank line between each, so a prompt can use it."""
    def __init__(self, read: str = "docs", write: str = "context"):
        self._read = read
        self._write = write

    def run(self, state: State) -> State:
        return {self._write: "\n\n".join(doc.page_content for doc in state[self._read])}


class Retry(Step):
    """Runs one step, trying again on the listed failures, up to `attempts` tries in all.
    Before retry n it waits wait * (2 ** (n - 1) + a random 0..1), so wait=0 never sleeps.
    """
    def __init__(self, step: Step, attempts: int = 3, on: tuple[type[Exception], ...] = (Exception,), wait: float = 1.0):
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}")
        self._step = step
        self._attempts = attempts
        self._on = on
        self._wait = wait

    def run(self, state: State) -> State:
        for attempt in range(1, self._attempts + 1):
            try:
                return self._step.run(state)
            except self._on:
                if attempt == self._attempts:
                    raise
                time.sleep(self._wait * (2 ** (attempt - 1) + random.uniform(0, 1)))
