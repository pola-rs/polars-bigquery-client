from __future__ import annotations

from collections.abc import Callable
from typing import Any

from polars_bigquery.core.compiler.ir.base import (
    Column,
    Expr,
    NullLiteral,
    Unsupported,
)

UNSUPPORTED_RECORD: tuple[str, Any, tuple[Any, ...]] = ("Leaf", Unsupported(), ())


def parse_null_literal(_value: Any) -> NullLiteral:
    """Parse a Null value from Polars JSON into a NullLiteral."""
    return NullLiteral()


# Keys represent Polars Rust AST `LiteralValue` / `AnyValue` enum variant names
# emitted inside `{"Literal": ...}` in serialized Polars JSON.
NULL_LITERAL_PARSERS: dict[str, Callable[[Any], Expr]] = {
    "Null": parse_null_literal,
}


def parse_column_expr(column_json: Any) -> tuple[str, Any, tuple[Any, ...]]:
    """Extract Column leaf node for a Polars `Column` node."""
    return ("Leaf", Column(name=str(column_json)), ())


def unwrap_literal_json(literal_json: Any) -> Any:
    """Unwrap Polars `Dyn`, `Scalar`, and `{"dtype": ..., "value": ...}` literal wrappers."""
    curr = literal_json
    while isinstance(curr, dict):
        if len(curr) == 1 and "Dyn" in curr:
            curr = curr["Dyn"]
        elif len(curr) == 1 and "Scalar" in curr:
            curr = curr["Scalar"]
        elif len(curr) == 2 and "dtype" in curr and "value" in curr:
            curr = curr["value"]
        else:
            break
    return curr
