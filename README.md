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
| 3 | Tools & function calling | Strategy pattern, tool registry | [`phase-3`](https://github.com/jairamshegde/mingraph/tree/phase-3) | [Strategy and the Registry Pattern Never Clicked for Me Until a Model Started Calling My Functions](https://thearchitectsmind.hashnode.dev/strategy-and-the-registry-pattern) | Done |
| 4 | Memory | Polymorphism, Template Method | [`phase-4`](https://github.com/jairamshegde/mingraph/tree/phase-4) | [Template Method, and Why Your LLM's Memory Isn't Its Chat History](https://thearchitectsmind.hashnode.dev/template-method-and-why-your-llm-s-memory-isn-t-its-chat-history) | Done |
| 5 | Steps | Composite pattern | [`phase-5`](https://github.com/jairamshegde/mingraph/tree/phase-5) | [The Composite Pattern in a Python LLM Pipeline](https://thearchitectsmind.hashnode.dev/the-composite-pattern-in-a-python-llm-pipeline) | Done |
| 6 | RAG retrievers | Dependency injection, interface segregation | [`phase-6`](https://github.com/jairamshegde/mingraph/tree/phase-6) | [Dependency Injection and Interface Segregation in a Python RAG Retriever](https://thearchitectsmind.hashnode.dev/dependency-injection-and-interface-segregation-in-a-python-rag-retriever) | Done |
| 7 | Agent loop | State, Observer (streaming and callbacks) | [`phase-7`](https://github.com/jairamshegde/mingraph/tree/phase-7) | [The State and Observer Patterns in a Python Agent Loop](https://thearchitectsmind.hashnode.dev/the-state-and-observer-patterns-in-a-python-agent-loop) | Done |
| ★ | Capstone | A mini multi-agent graph orchestrator | — | Coming soon | In progress |

## Project structure

```
mingraph/
├── mingraph/
│   ├── __init__.py
│   ├── agents.py       # Agent loop, its State cards, and the Observer, ConsoleLogger and TraceRecorder listeners
│   ├── llm.py          # BaseLLM contract and the LLMResponse it returns
│   ├── memory.py       # Memory base and the KeepAll, LastN and Summarise strategies
│   ├── messages.py     # Sealed message types, one per kind of line, and ToolCall
│   ├── prompts.py      # ChatPromptTemplate and MessagesPlaceholder
│   ├── providers.py    # OpenAI, Anthropic and Ollama adapters, and the Ollama embedder
│   ├── retrievers.py   # Document, Embeddings, VectorStore, Retriever and split_text
│   ├── steps.py        # Step, the Sequence/Branch/Parallel containers, wrapper steps, Retrieve and Retry
│   └── tools.py        # Tool, read off a function, and ToolRegistry
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
for example `qwen3` or a Gemma model such as `gemma3` or `gemma4`. The
retrieval example also needs an embedding model:

```bash
ollama pull qwen3-embedding:0.6b
```

## Usage

### Agent loop

An `Agent` calls the model, runs the tools it asks for, sends the results
back, and repeats until the model replies without asking for a tool. The model
decides how many rounds that takes; `max_steps` (10 by default) is the ceiling.
Listeners passed as `observers` hear every step as it happens, and the agent
runs the same with or without them:

```python
from mingraph.agents import Agent, ConsoleLogger, Observer, TraceRecorder
from mingraph.providers import OllamaLLM, OpenAILLM
from mingraph.tools import Tool, ToolRegistry


@Tool
def get_weather(city: str) -> str:
    """Get the current temperature for a city, in degrees Celsius."""
    return {"paris": "18°C", "tokyo": "24°C"}.get(city.lower(), "20°C")


@Tool
def to_fahrenheit(celsius: int) -> str:
    """Convert a temperature in degrees Celsius to degrees Fahrenheit."""
    return f"{round(celsius * 9 / 5 + 32)}°F"


class ToolCounter(Observer):
    """A listener overrides only the announcements it cares about."""
    def __init__(self):
        self.calls = 0

    def on_tool_call(self, call):
        self.calls += 1


registry = ToolRegistry()
registry.add(get_weather)
registry.add(to_fahrenheit)
counter = ToolCounter()

agent = Agent(
    OpenAILLM("gpt-5-mini", reasoning_effort="low"),  # or OllamaLLM("qwen3.5", think=True)
    registry,
    system_prompt="You are a travel assistant. Use the tools for temperatures and conversions.",
    observers=[ConsoleLogger(), TraceRecorder("trace.jsonl"), counter],
)
state = agent.run({"question": "What's the temperature in Paris and in Tokyo, in Fahrenheit?"})
print(state["answer"])    # e.g. Paris: 64°F, Tokyo: 75°F
print(counter.calls)      # e.g. 4: two lookups, two conversions
```

`ConsoleLogger` prints each step, including the model's thinking and token
counts when the provider reports them. `TraceRecorder` appends one JSON line
per step to a file, so a run that went wrong can be read afterwards. An
observer that raises is skipped with a logged warning and never stops the run.

The agent is a `Step`: it reads `question` and writes `answer`, so it fits
anywhere a step does, including inside a `Sequence`. It keeps the conversation
in a `KeepAll` memory unless you pass `memory=`, and the conversation carries
over to the next `run`. If the model calls a tool with arguments that don't
fit, or a tool that doesn't exist, the error goes back to the model so it can
correct the call. Any other tool failure stops the run. Running out of
`max_steps` raises `StepLimitReached` instead of returning a made-up answer.
Thinking is switched on per provider (`think=` for Ollama, `reasoning_effort=`
for OpenAI) and comes back on `LLMResponse.reasoning` and
`LLMResponse.reasoning_tokens`.

### Earlier phases

Each earlier example builds on the ones before it (imports, `llm`, `registry`,
`template` and `run` carry over), so open them in order.

<details>
<summary><strong>Chat with history (phases 1–2)</strong></summary>

A prompt template turns variables plus the conversation so far into a list of
messages. `generate` sends that list and returns the assistant's reply with
metadata about the call. Calling code depends only on `BaseLLM`, so switching
providers is a one-line change:

```python
from mingraph.llm import BaseLLM
from mingraph.messages import Message, SystemMessage, UserMessage
from mingraph.prompts import ChatPromptTemplate, MessagesPlaceholder
from mingraph.providers import AnthropicLLM, OllamaLLM, OpenAILLM

template = ChatPromptTemplate([
    SystemMessage("You are a helpful assistant. Answer in one short sentence."),
    MessagesPlaceholder("history"),
    UserMessage("{question}"),
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

</details>

<details>
<summary><strong>Tool calling (phase 3)</strong></summary>

`@Tool` turns a plain function into a tool. Its name, description and argument
schema are read off the function's name, docstring and type hints. A
`ToolRegistry` holds the tools, refuses two with the same name, and runs the
call the model asks for:

```python
from mingraph.tools import Tool, ToolRegistry


@Tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is 18°C and cloudy in {city}."


@Tool
def add(a: int, b: int) -> int:
    """Add two whole numbers."""
    return a + b


registry = ToolRegistry()
registry.add(get_weather)
registry.add(add)


def ask(llm: BaseLLM, question: str) -> str:
    messages: list[Message] = [
        SystemMessage("You are a helpful assistant. Use the tools when they help."),
        UserMessage(question),
    ]
    response = llm.generate(messages, tools=registry.tools)
    if response.stop_reason != "tool_call":
        return response.message.content

    messages.append(response.message)             # the model's request, with its tool calls
    for call in response.message.tool_calls:
        messages.append(registry.run(call))       # the result, carrying the call's id
    return llm.generate(messages, tools=registry.tools).message.content


llm = OpenAILLM("gpt-5-mini")  # or OllamaLLM("qwen3.5")
print(ask(llm, "What's the weather in Paris right now?"))
print(ask(llm, "What is 1234 + 5678?"))
```

The calling code never names a tool, so adding one is a new `@Tool` function
and one `registry.add` line. Tool arguments from the model are checked against
the function's signature before it runs. Tool calling works with OpenAI and
Ollama; `AnthropicLLM` raises `NotImplementedError` when given tools. The
example runs one round of tool calls; `Agent` (above) loops until the model
stops asking.

</details>

<details>
<summary><strong>Memory (phase 4)</strong></summary>

A `Memory` keeps the whole conversation and builds the list to send on each
call. The record only grows; a strategy decides how much of it the model sees.
`KeepAll` sends everything, `LastN` sends the last few turns, and `Summarise`
sends an LLM-written recap of older turns plus the recent ones. A turn is a
user message and everything after it up to the next one, so a tool call is
never separated from its result:

```python
from mingraph.memory import KeepAll, LastN, Memory, Summarise


def chat(llm: BaseLLM, memory: Memory, questions: list[str]) -> None:
    for question in questions:
        memory.add(UserMessage(question))
        response = llm.generate(memory.messages())
        memory.add(response.message)
    print(f"{response.message.content}  [{response.input_tokens} input tokens]")


system = SystemMessage("You help a family plan a trip. Answer in one short sentence.")
questions = ["Hi, I'm Asha. We land in Pune on the 5th.", "We're a family of four.",
             "Suggest one vegetarian dish to try there.", "What's my name, and which day do we land?"]

llm = OpenAILLM("gpt-5-mini")  # or OllamaLLM("qwen3.5")
chat(llm, KeepAll(system=system), questions)
chat(llm, LastN(1, system=system), questions)
chat(llm, Summarise(llm, keep=1, trigger=2, system=system), questions)
```

The calling code is the same for all three. `KeepAll` answers the last
question from the full history, `LastN(1)` can't answer it, and `Summarise`
answers from its recap for fewer input tokens. The system message is given
once, to the constructor, and always comes first. The recap is sent as a user
message and never stored in the conversation.

</details>

<details>
<summary><strong>Steps (phase 5)</strong></summary>

Every step has one method, `run(state)`. It reads what it needs from a shared
state dict and returns a dict of only the keys it set. The containers are
steps too: `Sequence` runs steps in order, `Branch` runs one step picked from
the state, and `Parallel` runs steps on the same state and merges their
results. So a whole pipeline fits anywhere a single step fits. `Prompt`,
`CallModel`, `Text` and `RunTools` wrap the template, the model and the tool
registry, and `Retry` wraps any one step:

```python
from mingraph.steps import Branch, CallModel, Parallel, Prompt, Retry, RunTools, Sequence, State, Step, Text

llm = OpenAILLM("gpt-5-mini")  # or OllamaLLM("qwen3.5")


def template(text: str) -> ChatPromptTemplate:
    return ChatPromptTemplate([UserMessage(text)])


# Two model calls in a row: draft a sentence, then shorten it.
tagline = Sequence([
    Prompt(template("Write one sentence about a family trip to {city}."), write="draft_msgs"),
    Retry(CallModel(llm, read="draft_msgs", write="draft_reply")),
    Text(read="draft_reply", write="draft"),
    Prompt(template("Shorten this to at most six words: {draft}"), write="short_msgs"),
    CallModel(llm, read="short_msgs", write="tagline"),
])

# One model call that may ask for a tool, and a Branch that runs it if it did.
weather = Sequence([
    Prompt(template("What's the weather in {city} right now?")),
    CallModel(llm, tools=registry.tools),
    Branch(lambda state: "tool" if state["reply"].tool_calls else "done",
           {"tool": RunTools(registry), "done": Sequence([])}),
])

trip = Parallel([tagline, weather])


def run(step: Step, start: State) -> State:
    return {**start, **step.run(start)}


state = run(trip, {"city": "Pune"})
print(state["tagline"].content)                                # e.g. Family weekend: Shaniwar Wada, vada-pav, sunset
print([result.content for result in state["tool_results"]])    # ['It is 18°C and cloudy in Pune.']
```

The imports and `registry` come from the examples above. `run` never checks what it was given. `run(weather, {"city": "Pune"})` works
the same way as `run(trip, ...)`, as would a single `CallModel`. `Text` is
there because `CallModel` stores the reply message, and a template would print
the whole message object instead of its text. `Parallel` raises a `ValueError`
if two of its steps set the same key, which is why the two pipelines use
different slot names. `Retry` tries a failing step up to three times, waiting
longer each time, and wrapping only the step that can fail keeps the steps
before it from running twice. The steps inside `Parallel` run one after
another, not at the same time.

</details>

<details>
<summary><strong>Retrieval, RAG (phase 6)</strong></summary>

Retrieval finds the documents that match a question and puts them in the
prompt. Three small interfaces split the job. An `Embeddings` model turns text
into a vector, so texts with similar meaning land close together. A
`VectorStore` keeps documents with their vectors and finds the ones nearest a
query. A `Retriever` only returns documents for a query. The store is handed
its embedder, and the `Retrieve` step is handed a retriever, never the store:

```python
from mingraph.providers import OllamaEmbedder
from mingraph.retrievers import Document, InMemoryVectorStore, VectorStoreRetriever
from mingraph.steps import FormatDocs, Retrieve

cards = [
    "Jairam's dal: 1 cup toor, pressure cook for 3 whistles, then tadka with ghee and hing.",
    "Rajma: soak overnight, pressure cook for 6 whistles until soft.",
    "Gulab jamun: fry the balls on low heat till golden, then soak in warm sugar syrup.",
]
store = InMemoryVectorStore(OllamaEmbedder("qwen3-embedding:0.6b", dimensions=384))
store.add_documents([Document(card, {"source": "family recipes"}) for card in cards])

rag = Sequence([
    Retrieve(VectorStoreRetriever(store, k=2)),
    FormatDocs(),
    Prompt(template("Answer using only these recipes. If they don't say, say so.\n\n{context}\n\nQuestion: {question}")),
    CallModel(llm),
])

state = run(rag, {"question": "How many whistles for Jairam's dal?"})
print(state["reply"].content)                          # 3 whistles.
print([doc.metadata["source"] for doc in state["docs"]])
```

`llm`, `template`, `run` and the step imports come from the examples above.
Swapping the embedder touches only the line that builds the store: a test can
hand it a fake `Embeddings` that returns vectors picked by hand, so the
retrieval order is known in advance. One embedder makes every vector in a
store, cards and questions alike. The store ranks by cosine similarity and
checks every document on each search. `OllamaEmbedder` takes a `query_prefix`
for models that want an instruction in front of questions, and
`split_text(text, chunk_size, chunk_overlap)` cuts a long file into pieces to
add as documents. The splitter cuts every `chunk_size` characters without
reading the text, so a piece can lose the words that say what it's about.

</details>

## References

- [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview)
- [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [Python `abc` module](https://docs.python.org/3/library/abc.html)
- [Python `dataclasses` module](https://docs.python.org/3/library/dataclasses.html)
- [LangChain `ChatPromptTemplate` reference](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate)
- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain `Runnable` reference](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangChain retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)
- [LangChain vector stores](https://docs.langchain.com/oss/python/integrations/vectorstores)
- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain streaming](https://docs.langchain.com/oss/python/langchain/streaming)
- [LangChain observability](https://docs.langchain.com/oss/python/langchain/observability)
- [LangChain short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [OpenAI cookbook: session memory](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)
- [Python `inspect` module](https://docs.python.org/3/library/inspect.html)
- [Refactoring.Guru: design patterns](https://refactoring.guru/design-patterns)
- Provider SDKs: [openai-python](https://github.com/openai/openai-python),
  [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python),
  [ollama-python](https://github.com/ollama/ollama-python)

## License

Released under the [MIT License](LICENSE).
