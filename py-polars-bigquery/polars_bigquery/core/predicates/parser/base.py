from __future__ import annotations

from collections.abc import Callable
from typing import Any

from polars_bigquery.core.predicates.ir.base import (
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
