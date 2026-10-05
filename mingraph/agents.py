import logging
import warnings
from collections.abc import Sequence as Seq

from mingraph.llm import BaseLLM, LLMResponse
from mingraph.memory import KeepAll, Memory
from mingraph.messages import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from mingraph.steps import State, Step
from mingraph.tools import ToolArgsError, ToolRegistry

logger = logging.getLogger(__name__)


class StepLimitReached(Exception):
    """The model was still asking for tools when the agent ran out of model calls."""


class Observer:
    """Hears what an agent does, as it happens. Override only the announcements you care about.
    Observers only watch: nothing they do changes the run, and an observer that raises is skipped.
    """
    def on_step_start(self, step: int) -> None:
        """Before each model call, numbered from 1."""

    def on_model_reply(self, response: LLMResponse) -> None:
        """After each model call: the reply plus tokens, stop reason and any thinking."""

    def on_tool_call(self, call: ToolCall) -> None:
        """Before a tool runs."""

    def on_tool_return(self, result: ToolMessage) -> None:
        """After a tool ran and its answer was recorded."""

    def on_tool_error(self, call: ToolCall, error: Exception) -> None:
        """A tool call failed: sent back to the model if it was the model's mistake, otherwise the run stops."""

    def on_finish(self, answer: str | None, reason: str) -> None:
        """Once per run: reason is "answered", or "step limit" with no answer."""


class Agent(Step):
    """Calls the model, runs the tools it asks for, feeds the answers back, and repeats until it answers.
    Reads a question from one slot and writes the final answer's text to another, so it fits anywhere a step does.
    The conversation lives in its memory and carries over between runs.
    """
    def __init__(self, llm: BaseLLM, registry: ToolRegistry, system_prompt: str, memory: Memory | None = None,
                 observers: Seq[Observer] = (), max_steps: int = 10, read: str = "question", write: str = "answer"):
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        card = SystemMessage(system_prompt)
        if memory is None:
            memory = KeepAll(card)   # built here, never as a default argument, so agents never share one
        elif memory.system != card:
            warnings.warn("memory's system message was replaced by the agent's system_prompt", stacklevel=2)
            memory.system = card
        self._llm = llm
        self._registry = registry
        self._memory = memory
        self._observers = tuple(observers)
        self._max_steps = max_steps
        self._read = read
        self._write = write

    def run(self, state: State) -> State:
        self._memory.add(UserMessage(state[self._read]))
        card = _CallingModel(step=1)
        while card is not None:
            last, card = card, card.run(self)   # do what the card says; it hands back the next one
        return {self._write: last.answer}       # the last card is always _Finished

    def _announce(self, event: str, *args) -> None:
        for observer in self._observers:
            try:
                getattr(observer, event)(*args)
            except Exception:
                logger.warning("observer %s failed on %s", type(observer).__name__, event, exc_info=True)


# --- the cards: each does one situation's work and returns the next card ---

class _CallingModel:
    def __init__(self, step: int):
        self.step = step

    def run(self, agent: Agent):
        if self.step > agent._max_steps:
            return _Finished(None, "step limit")
        agent._announce("on_step_start", self.step)
        response = agent._llm.generate(agent._memory.messages(), agent._registry.tools)
        agent._memory.add(response.message)
        agent._announce("on_model_reply", response)
        if response.message.tool_calls:
            return _RunningTools(response.message, self.step)
        return _Finished(response.message.content, "answered")


class _RunningTools:
    def __init__(self, reply: AssistantMessage, step: int):
        self.reply = reply
        self.step = step

    def run(self, agent: Agent):
        for call in self.reply.tool_calls:
            agent._announce("on_tool_call", call)
            try:
                result = agent._registry.run(call)
            except ToolArgsError as e:
                # The model's mistake: it gets the error as the answer and can send a corrected call.
                agent._memory.add(ToolMessage(f"Error: {e}\n Please fix your mistakes.", tool_call_id=call.id))
                agent._announce("on_tool_error", call, e)
                continue
            except Exception as e:
                # Not fixable by the model: answer the slip so memory stays valid for the next run, then stop.
                agent._memory.add(ToolMessage(f"Error: tool failed: {e}", tool_call_id=call.id))
                agent._announce("on_tool_error", call, e)
                raise
            agent._memory.add(result)
            agent._announce("on_tool_return", result)
        return _CallingModel(self.step + 1)


class _Finished:
    def __init__(self, answer: str | None, reason: str):
        self.answer = answer
        self.reason = reason

    def run(self, agent: Agent):
        agent._announce("on_finish", self.answer, self.reason)
        if self.answer is None:
            raise StepLimitReached(f"no answer after {agent._max_steps} model calls")
        return None
