from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Contract every LLM provider must satisfy.
    Callers depend only on this interface, never on a specific provider.
    """
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to the model and return its text response."""
        raise NotImplementedError
