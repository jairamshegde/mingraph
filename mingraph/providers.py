from anthropic import Anthropic
from ollama import Client as OllamaClient
from openai import OpenAI

from mingraph.llm import BaseLLM, LLMResponse
from mingraph.messages import AssistantMessage, Message, ToolMessage


class OpenAILLM(BaseLLM):
    """OpenAI adapter for BaseLLM, backed by the Chat Completions API."""

    _STOP_REASONS = {"stop": "stop", "length": "length"}

    def __init__(self, model: str, api_key: str | None = None):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, messages: list[Message]) -> LLMResponse:
        _reject_tool_lines(messages)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            message=AssistantMessage(choice.message.content or ""),
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            stop_reason=self._STOP_REASONS.get(choice.finish_reason, "other"),
        )


class OllamaLLM(BaseLLM):
    """Local-model adapter for BaseLLM, backed by a running Ollama server."""

    _STOP_REASONS = {"stop": "stop", "length": "length"}

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self._client = OllamaClient(host=host)
        self._model = model

    def generate(self, messages: list[Message]) -> LLMResponse:
        _reject_tool_lines(messages)
        response = self._client.chat(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return LLMResponse(
            message=AssistantMessage(response.message.content or ""),
            input_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count,
            stop_reason=self._STOP_REASONS.get(response.done_reason, "other"),
        )


class AnthropicLLM(BaseLLM):
    """Anthropic adapter for BaseLLM, backed by the Messages API."""

    _STOP_REASONS = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
    }

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024):
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, messages: list[Message]) -> LLMResponse:
        _reject_tool_lines(messages)
        # The Messages API has no "system" role; system prompts go in the top-level `system` param.
        request = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages if m.role != "system"],
        }
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        if system:
            request["system"] = system
        response = self._client.messages.create(**request)
        return LLMResponse(
            message=AssistantMessage("".join(b.text for b in response.content if b.type == "text")),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=self._STOP_REASONS.get(response.stop_reason, "other"),
        )


def _reject_tool_lines(messages: list[Message]) -> None:
    for m in messages:
        if isinstance(m, ToolMessage) or (isinstance(m, AssistantMessage) and m.tool_calls):
            raise NotImplementedError("tool calls and results aren't translated for providers yet")
