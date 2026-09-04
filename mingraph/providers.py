from anthropic import Anthropic
from openai import OpenAI

from mingraph.llm import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI adapter for BaseLLM, backed by the Responses API."""

    def __init__(self, model: str, api_key: str | None = None):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.responses.create(model=self._model, input=prompt)
        return response.output_text


class AnthropicLLM(BaseLLM):
    """Anthropic adapter for BaseLLM, backed by the Messages API."""

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024):
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
