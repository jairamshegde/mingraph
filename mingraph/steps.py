from abc import ABC, abstractmethod

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
