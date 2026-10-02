import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Document:
    """A piece of text that can be retrieved, plus facts about it (e.g. which file it came from).
    Metadata is copied and kept read-only, so a document can't change after it's made.
    """
    page_content: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class Embeddings(ABC):
    """Turns text into a vector, so texts with similar meaning land close together.
    Documents and queries have separate calls because some models treat them differently.
    """
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """The vector for a search query."""
        raise NotImplementedError


class VectorStore(ABC):
    """Stores documents by their vectors and finds the ones nearest to a query."""
    @abstractmethod
    def add_documents(self, docs: list[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query: str, k: int) -> list[Document]:
        """The k documents nearest to the query, nearest first."""
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """Keeps (document, vector) pairs in a list and checks every one on each search.
    Handed one embedder, used for both documents and queries, so every vector is on the same map.
    """
    def __init__(self, embedder: Embeddings):
        self._embedder = embedder
        self._entries: list[tuple[Document, list[float]]] = []

    def add_documents(self, docs: list[Document]) -> None:
        vectors = self._embedder.embed_documents([doc.page_content for doc in docs])
        self._entries.extend(zip(docs, vectors, strict=True))

    def similarity_search(self, query: str, k: int) -> list[Document]:
        q = self._embedder.embed_query(query)
        ranked = sorted(self._entries, key=lambda entry: _cosine(q, entry[1]), reverse=True)
        return [doc for doc, _ in ranked[:k]]


class Retriever(ABC):
    """Returns documents for a query. That's all a caller of a retriever can do."""
    @abstractmethod
    def invoke(self, query: str) -> list[Document]:
        raise NotImplementedError


class VectorStoreRetriever(Retriever):
    """A retriever backed by a vector store, returning a fixed number of documents."""
    def __init__(self, store: VectorStore, k: int = 4):
        self._store = store
        self._k = k

    def invoke(self, query: str) -> list[Document]:
        return self._store.similarity_search(query, self._k)


def _cosine(a: list[float], b: list[float]) -> float:
    # Compares direction only: 1 when two vectors point the same way, 0 at a right angle.
    return sum(x * y for x, y in zip(a, b, strict=True)) / (math.hypot(*a) * math.hypot(*b))
