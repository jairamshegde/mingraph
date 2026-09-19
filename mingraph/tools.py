import copy
import inspect
from collections.abc import Callable, Mapping

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


class Tool:
    """A function the model can be offered and can ask to call.
    Name, description and argument schema are read off the function, so a tool is declared in one place.
    Use as a decorator: `@Tool` above a function with a docstring and str/int/float/bool parameters.
    """

    def __init__(self, fn: Callable):
        if not fn.__doc__:
            raise ValueError(f"tool {fn.__name__!r} needs a docstring; it becomes the description the model reads")
        self._fn = fn
        self._signature = inspect.signature(fn)
        properties, required = {}, []
        for p in self._signature.parameters.values():
            if p.annotation not in _JSON_TYPES:
                got = "no type" if p.annotation is inspect.Parameter.empty else repr(p.annotation)
                raise TypeError(f"parameter {p.name!r} of tool {fn.__name__!r} must be str, int, float or bool, got {got}")
            properties[p.name] = {"type": _JSON_TYPES[p.annotation]}
            if p.default is inspect.Parameter.empty:
                required.append(p.name)
        self._parameters = {"type": "object", "properties": properties, "required": required}

    @property
    def name(self) -> str:
        return self._fn.__name__

    @property
    def description(self) -> str:
        return inspect.getdoc(self._fn)

    @property
    def parameters(self) -> dict:
        """JSON Schema for the arguments. A copy, so changing it can't change the tool."""
        return copy.deepcopy(self._parameters)

    def run(self, args: Mapping[str, object]) -> str:
        """Check the arguments against the function's signature, call it, and return the result as text."""
        try:
            bound = self._signature.bind(**args)
        except TypeError as e:
            raise TypeError(f"tool {self.name!r}: {e}") from None
        for name, value in bound.arguments.items():
            expected = self._signature.parameters[name].annotation
            if not _matches(value, expected):
                raise TypeError(f"tool {self.name!r}: {name!r} must be {expected.__name__}, got {type(value).__name__}")
        return str(self._fn(**args))


def _matches(value: object, expected: type) -> bool:
    if expected is float:
        return type(value) in (int, float)  # JSON has one number type, so 3 is a valid float
    return type(value) is expected  # strict: True is an int in Python, but not here
