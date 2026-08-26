"""Parsing of the literals carried inside the simulator's `extra_params` column.

`current_inventory` arrives as a string holding a Python literal, e.g.
"[[12, 5], [40, 3]]" for (shelf life day, quantity) pairs. It must stay a
literal parse: the column is fully caller-controlled on the simulation
endpoints, so evaluating it as an expression would hand any caller code
execution inside the container.
"""
import ast
from typing import Any


def parse_inventory_literal(value: Any) -> Any:
    """Return `value` parsed as a Python literal, or unchanged if not a string."""
    if value is None:
        return []
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return []

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError(
            f"current_inventory is not a valid literal: {text[:120]!r}"
        ) from exc
