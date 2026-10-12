# mingraph

LangChain and LangGraph building blocks, rebuilt from scratch to learn
object-oriented design and design patterns in Python.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> *What I cannot create, I do not understand.* (Richard Feynman)

Each phase rebuilds one piece of the LangChain/LangGraph stack and uses it to
learn one OOP idea or pattern. It's a learning project in the spirit of
[build-your-own-x](https://github.com/codecrafters-io/build-your-own-x), not a
library and not production code.

## Learning series

One post per phase on
[The Architect's Mind](https://jairamshegde.github.io/thearchitectsmind/).
The blog is moving off Hashnode with the same name. Phases 0 and 1 have moved,
and the rest follow.

| # | Phase | OOP idea | Code | Post |
|---|-------|----------|------|------|
| 0 | Why OOP clicked | Design problem first | | [Learning OOP through the design questions I already ask at work](https://jairamshegde.github.io/thearchitectsmind/writing/phase-0-oop-finally-clicked/) |
| 1 | Provider wrapper | Abstraction, polymorphism | [`phase-1`](https://github.com/jairamshegde/mingraph/tree/phase-1) | [Abstraction and polymorphism, built from three LLM adapters](https://jairamshegde.github.io/thearchitectsmind/writing/phase-1-abstraction-and-polymorphism/) |
| 2 | Messages & prompts | Encapsulation, composition, dataclasses | [`phase-2`](https://github.com/jairamshegde/mingraph/tree/phase-2) | [What Encapsulation Actually Buys You](https://thearchitectsmind.hashnode.dev/what-encapsulation-actually-buys-you) |
| 3 | Tools | Strategy, registry | [`phase-3`](https://github.com/jairamshegde/mingraph/tree/phase-3) | [Strategy and the Registry Pattern](https://thearchitectsmind.hashnode.dev/strategy-and-the-registry-pattern) |
| 4 | Memory | Template Method | [`phase-4`](https://github.com/jairamshegde/mingraph/tree/phase-4) | [Template Method, and Why Your LLM's Memory Isn't Its Chat History](https://thearchitectsmind.hashnode.dev/template-method-and-why-your-llm-s-memory-isn-t-its-chat-history) |
| 5 | Steps | Composite | [`phase-5`](https://github.com/jairamshegde/mingraph/tree/phase-5) | [The Composite Pattern in a Python LLM Pipeline](https://thearchitectsmind.hashnode.dev/the-composite-pattern-in-a-python-llm-pipeline) |
| 6 | RAG retrievers | Dependency injection, interface segregation | [`phase-6`](https://github.com/jairamshegde/mingraph/tree/phase-6) | [Dependency Injection and Interface Segregation in a Python RAG Retriever](https://thearchitectsmind.hashnode.dev/dependency-injection-and-interface-segregation-in-a-python-rag-retriever) |
| 7 | Agent loop | State, Observer | [`phase-7`](https://github.com/jairamshegde/mingraph/tree/phase-7) | [The State and Observer Patterns in a Python Agent Loop](https://thearchitectsmind.hashnode.dev/the-state-and-observer-patterns-in-a-python-agent-loop) |
| ★ | Capstone | Mini multi-agent graph orchestrator | | In progress |

## Project structure

```
mingraph/
├── mingraph/
│   ├── agents.py       # Agent loop, State cards, observers
│   ├── llm.py          # BaseLLM contract, LLMResponse
│   ├── memory.py       # Memory base; KeepAll, LastN, Summarise
│   ├── messages.py     # one sealed type per message kind, ToolCall
│   ├── prompts.py      # ChatPromptTemplate, MessagesPlaceholder
│   ├── providers.py    # OpenAI, Anthropic, Ollama adapters; Ollama embedder
│   ├── retrievers.py   # Document, Embeddings, VectorStore, Retriever, split_text
│   ├── steps.py        # Step; Sequence, Branch, Parallel; Retrieve, Retry
│   └── tools.py        # Tool, ToolRegistry
└── requirements.txt    # pinned provider SDKs
```

## Getting started

You need Python 3.10+, plus an OpenAI or Anthropic key, or
[Ollama](https://ollama.com/download) running locally.

```bash
git clone https://github.com/jairamshegde/mingraph.git
cd mingraph
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="..."        # or pass api_key= to the provider
export ANTHROPIC_API_KEY="..."
ollama pull qwen3.5                # any Ollama chat model works
ollama pull qwen3-embedding:0.6b   # only for the RAG example
```

Run code from the repo root. The project isn't packaged yet.

## Usage

### Agent loop

The agent calls the model, runs the tools it asks for, and repeats until the
model answers without a tool call.

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

- `max_steps` (default 10) caps the loop. Hitting it raises `StepLimitReached`.
- `ConsoleLogger` prints each step. `TraceRecorder` writes one JSON line per step.
- A failing observer is skipped with a warning and never stops the run.
- Bad tool arguments or unknown tools go back to the model to fix. Other tool errors stop the run.
- The agent is a `Step` (reads `question`, writes `answer`), and it keeps a `KeepAll` memory unless you pass `memory=`.
- Thinking is `think=` for Ollama and `reasoning_effort=` for OpenAI. It comes back on `LLMResponse.reasoning`.

### Earlier phases

Open these in order. Each one reuses imports and names from the ones before it.

<details>
<summary><strong>Chat with history (phases 1 and 2)</strong></summary>

Calling code depends only on `BaseLLM`, so switching providers is a one-line change.

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

- The second answer can only come from the history.
- A missing or unknown template variable raises an error.
- With a thinking model, `output_tokens` includes hidden reasoning.

</details>

<details>
<summary><strong>Tool calling (phase 3)</strong></summary>

`@Tool` reads a tool's name, description and schema off the function. `ToolRegistry` refuses duplicate names and runs the calls.

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

- Adding a tool means one `@Tool` function and one `registry.add` line.
- Arguments are checked against the function signature before it runs.
- Works with OpenAI and Ollama. `AnthropicLLM` raises `NotImplementedError` for tools.

</details>

<details>
<summary><strong>Memory (phase 4)</strong></summary>

`Memory` keeps the whole conversation. A strategy picks what the model sees: `KeepAll` sends everything, `LastN` sends the last turns, and `Summarise` sends a recap plus recent turns.

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

- `KeepAll` answers the last question. `LastN(1)` can't. `Summarise` can, with fewer input tokens.
- Trimming works by turns, so a tool call never loses its result.
- The system message goes to the constructor and always comes first.

</details>

<details>
<summary><strong>Steps (phase 5)</strong></summary>

Every step has one method, `run(state)`, and returns only the keys it set. Containers are steps too, so a whole pipeline fits wherever one step does.

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

- `Sequence` runs in order, `Branch` picks one step, `Parallel` merges results.
- `Parallel` raises `ValueError` if two steps set the same key. Its steps run one after another.
- `Retry` retries one step up to three times with growing waits.
- `Text` pulls the text out of a reply message so a template can use it.

</details>

<details>
<summary><strong>Retrieval, RAG (phase 6)</strong></summary>

An `Embeddings` model turns text into vectors, a `VectorStore` finds the nearest documents, and a `Retriever` returns them for a query.

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

- The store is handed its embedder, so a test can pass a fake one.
- One embedder makes every vector in a store. Search ranks by cosine similarity.
- `split_text` cuts long text into fixed-size chunks, which can split a topic in half.

</details>

## References

- [LangChain docs](https://docs.langchain.com/oss/python/langchain/overview) and [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview)
- Python: [`abc`](https://docs.python.org/3/library/abc.html), [`dataclasses`](https://docs.python.org/3/library/dataclasses.html), [`inspect`](https://docs.python.org/3/library/inspect.html)
- [Refactoring.Guru: design patterns](https://refactoring.guru/design-patterns)
- Provider SDKs: [openai-python](https://github.com/openai/openai-python), [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python), [ollama-python](https://github.com/ollama/ollama-python)

<details>
<summary>More LangChain pages used per phase</summary>

- [`ChatPromptTemplate`](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate), [tools](https://docs.langchain.com/oss/python/langchain/tools), [`Runnable`](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable), [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval), [vector stores](https://docs.langchain.com/oss/python/integrations/vectorstores), [agents](https://docs.langchain.com/oss/python/langchain/agents), [streaming](https://docs.langchain.com/oss/python/langchain/streaming)
- [Observability](https://docs.langchain.com/oss/python/langchain/observability), [short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory), [OpenAI cookbook: session memory](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)

</details>

## License

[MIT](LICENSE)
