# mingraph

LangChain and LangGraph building blocks, rebuilt from scratch to learn
object-oriented design and design patterns in Python.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Inspiration

> *What I cannot create, I do not understand.* (Richard Feynman)

mingraph follows the idea behind
[build-your-own-x](https://github.com/codecrafters-io/build-your-own-x): the
best way to understand a technology is to rebuild it from scratch. Here the
goal is object-oriented design. Instead of studying patterns in isolation,
each one is learned by rebuilding a LangChain/LangGraph piece that actually
needs it, so every concept comes with a real problem it solves.

## About

mingraph is a learning project, not a library. Each phase takes one piece of
the LangChain/LangGraph stack (provider wrappers, messages, tools, memory,
steps, retrievers, the agent loop), rebuilds a minimal version of it, and uses
it to practise one object-oriented idea or design pattern. The roadmap ends
with a capstone: a tiny multi-agent graph orchestrator.

Design choices are checked against the official LangChain, LangGraph and
provider documentation. The reasoning behind each phase, including what went
wrong, is written up in the blog series below.

**What it isn't:** a replacement for LangChain or LangGraph, or
production-ready code. The goal is to understand why those frameworks are
shaped the way they are.

## Learning series

### LangGraph from Scratch: Design Patterns in Python

One post per phase on
[The Architect's Mind](https://thearchitectsmind.hashnode.dev/), covering what
was built, the design decisions behind it, and what went wrong along the way.

| # | Phase | OOP concept / pattern | Code | Blog post | Status |
|---|-------|-----------------------|------|-----------|--------|
| 1 | Provider wrapper | Abstraction, polymorphism | [`phase-1`](https://github.com/jairamshegde/mingraph/tree/phase-1) | [Abstraction and Polymorphism Never Clicked for Me Until I Wrapped Three LLM APIs](https://thearchitectsmind.hashnode.dev/abstraction-and-polymorphism-never-clicked-for-me-until-i-wrapped-three-llm-apis) | Done |
| 2 | Messages & prompts | Encapsulation, composition, dataclasses | [`phase-2`](https://github.com/jairamshegde/mingraph/tree/phase-2) | [What Encapsulation Actually Buys You](https://thearchitectsmind.hashnode.dev/what-encapsulation-actually-buys-you) | Done |
| 3 | Tools & function calling | Strategy pattern, tool registry | — | Coming soon | In progress |
| 4 | Memory | Polymorphism, Template Method | — | Coming soon | Planned |
| 5 | Steps | Composite pattern | — | Coming soon | Planned |
| 6 | RAG retrievers | Dependency injection, interface segregation | — | Coming soon | Planned |
| 7 | Agent loop | State, Observer (streaming and callbacks) | — | Coming soon | Planned |
| ★ | Capstone | A mini multi-agent graph orchestrator | — | Coming soon | Planned |

## Project structure

```
mingraph/
├── mingraph/
│   ├── __init__.py
│   ├── llm.py          # BaseLLM contract and the LLMResponse it returns
│   ├── messages.py     # Message: a sealed, validated role + content
│   ├── prompts.py      # ChatPromptTemplate and MessagesPlaceholder
│   └── providers.py    # OpenAI, Anthropic and Ollama adapters
├── requirements.txt    # pinned provider SDKs
├── LICENSE
└── README.md
```

## Getting started

### Prerequisites

- Python 3.10 or newer (developed on 3.12)
- An OpenAI and/or Anthropic API key, for the hosted providers
- [Ollama](https://ollama.com/download) running locally, for the local provider

### Installation

```bash
git clone https://github.com/jairamshegde/mingraph.git
cd mingraph
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project isn't packaged yet, so run code from the repository root so that
`import mingraph` resolves.

### Configuration

The OpenAI and Anthropic SDKs read their keys from the environment:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

You can also pass `api_key=` to the provider's constructor. Ollama needs no
key: `OllamaLLM` connects to `http://localhost:11434` unless you pass `host=`.
Pull a model before first use:

```bash
ollama pull qwen3.5
```

Any chat model from the [Ollama library](https://ollama.com/library) works,
for example `qwen3` or a Gemma model such as `gemma3` or `gemma4`.

## Usage

A prompt template turns variables plus the conversation so far into a list of
messages. `generate` sends that list and returns the assistant's reply with
metadata about the call. Calling code depends only on `BaseLLM`, so switching
providers is a one-line change:

```python
from mingraph.llm import BaseLLM
from mingraph.messages import Message
from mingraph.prompts import ChatPromptTemplate, MessagesPlaceholder
from mingraph.providers import AnthropicLLM, OllamaLLM, OpenAILLM

template = ChatPromptTemplate([
    Message("system", "You are a helpful assistant. Answer in one short sentence."),
    MessagesPlaceholder("history"),
    Message("user", "{question}"),
])


def chat(llm: BaseLLM, questions: list[str]) -> None:
    history: list[Message] = []
    for question in questions:
        messages = template.format_messages(history=history, question=question)
        response = llm.generate(messages)
        print(f"{response.message.content}  [{response.stop_reason}, {response.output_tokens} output tokens]")
        history += [messages[-1], response.message]


llm = OllamaLLM("qwen3.5")  # or OpenAILLM("gpt-5-mini"), AnthropicLLM("claude-sonnet-5")
chat(llm, ["Hi, I'm Jai.", "What's my name?"])
```

The second answer can only come from the history. Filling a template fails
loudly on a missing or unknown variable, and the first turn passes `history=[]`
explicitly. With a thinking model such as `qwen3.5`, `output_tokens` includes
the model's hidden reasoning, so it can be far larger than the visible reply.

## References

- [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview)
- [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [Python `abc` module](https://docs.python.org/3/library/abc.html)
- [Python `dataclasses` module](https://docs.python.org/3/library/dataclasses.html)
- [LangChain `ChatPromptTemplate` reference](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate)
- [Refactoring.Guru: design patterns](https://refactoring.guru/design-patterns)
- Provider SDKs: [openai-python](https://github.com/openai/openai-python),
  [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python),
  [ollama-python](https://github.com/ollama/ollama-python)

## License

Released under the [MIT License](LICENSE).
