import json
import logging
import warnings
from collections.abc import Sequence as Seq
from datetime import datetime
from os import PathLike

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


class ConsoleLogger(Observer):
    """Prints every announcement as it happens, thinking and token counts included. A count shows ? when
    the provider didn't report it.
    """
    def on_step_start(self, step: int) -> None:
        print(f"[step {step}]")

    def on_model_reply(self, response: LLMResponse) -> None:
        if response.reasoning:
            print(f"  thinking: {response.reasoning}")
        message = response.message
        said = f"(asks for {', '.join(c.name for c in message.tool_calls)})" if message.tool_calls else repr(message.content)
        print(f"  reply: {said}   tokens in={_count(response.input_tokens)} out={_count(response.output_tokens)}"
              f" thinking={_count(response.reasoning_tokens)}")

    def on_tool_call(self, call: ToolCall) -> None:
        print(f"  call {call.name} {dict(call.args)}")

    def on_tool_return(self, result: ToolMessage) -> None:
        print(f"  returned: {result.content}")

    def on_tool_error(self, call: ToolCall, error: Exception) -> None:
        print(f"  failed {call.name}: {type(error).__name__}: {error}")

    def on_finish(self, answer: str | None, reason: str) -> None:
        print(f"[finish: {reason}] {answer if answer is not None else ''}".rstrip())


class TraceRecorder(Observer):
    """Appends every announcement to a file, one JSON object per line, so a run can be read after it ends.
    The file is opened and closed for each line, so a run that crashes keeps everything written before it.
    """
    def __init__(self, path: str | PathLike):
        self._path = path
        self._run = 0
        self._step = 0

    def on_step_start(self, step: int) -> None:
        if step == 1:
            self._run += 1
        self._step = step
        self._write("step_start")

    def on_model_reply(self, response: LLMResponse) -> None:
        self._write("model_reply", content=response.message.content,
                    tool_calls=[{"name": c.name, "args": dict(c.args), "id": c.id} for c in response.message.tool_calls],
                    reasoning=response.reasoning, input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens, reasoning_tokens=response.reasoning_tokens,
                    stop_reason=response.stop_reason)

    def on_tool_call(self, call: ToolCall) -> None:
        self._write("tool_call", name=call.name, args=dict(call.args), id=call.id)

    def on_tool_return(self, result: ToolMessage) -> None:
        self._write("tool_return", content=result.content, tool_call_id=result.tool_call_id)

    def on_tool_error(self, call: ToolCall, error: Exception) -> None:
        self._write("tool_error", name=call.name, id=call.id, error=f"{type(error).__name__}: {error}")

    def on_finish(self, answer: str | None, reason: str) -> None:
        self._write("finish", answer=answer, reason=reason)

    def _write(self, event: str, **fields) -> None:
        line = {"time": datetime.now().isoformat(timespec="seconds"), "run": self._run, "step": self._step,
                "event": event, **fields}
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _count(n: int | None) -> str:
    return "?" if n is None else str(n)


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
