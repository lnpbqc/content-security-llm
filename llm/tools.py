"""Basic callable tools and a discoverable tool registry."""

from typing import List

from langchain_core.tools import BaseTool, tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide one integer by another."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def get_all_tools() -> List[BaseTool]:
    """Return every module-level function decorated with @tool."""
    found = {value.name: value for value in globals().values() if isinstance(value, BaseTool)}
    return [found[name] for name in sorted(found)]
