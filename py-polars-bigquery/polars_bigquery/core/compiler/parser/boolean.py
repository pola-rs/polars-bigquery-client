from __future__ import annotations

from collections.abc import Callable
from typing import Any

from polars_bigquery.core.compiler.ir.base import (
    BinaryExpr,
    Expr,
    UnaryExpr,
)
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    IsNotNull,
    IsNull,
    Not,
    Or,
)
from polars_bigquery.core.compiler.parser.base import UNSUPPORTED_RECORD


def parse_bool_literal(value: Any) -> BoolLiteral:
    """Parse a Boolean value from Polars JSON into a BoolLiteral."""
    return BoolLiteral(value=bool(value))


# Keys represent Polars Rust AST `LiteralValue` / `AnyValue` enum variant names
# emitted inside `{"Literal": ...}` in serialized Polars JSON.
BOOLEAN_LITERAL_PARSERS: dict[str, Callable[[Any], Expr]] = {
    "Boolean": parse_bool_literal,
}

# Keys represent Polars Rust AST `Operator` enum variant names emitted under
# `{"BinaryExpr": {"op": "<key>"}}` when serializing `pl.Expr.meta.serialize(format="json")`.
LOGICAL_BINARY_OPS: dict[str, type[BinaryExpr]] = {
    "And": And,
    "Or": Or,
}

# Keys represent Polars Rust AST `BooleanFunction` enum variant names emitted under
# `{"Function": {"function": {"Boolean": "<key>"}}}` in serialized Polars JSON.
BOOLEAN_UNARY_OPS: dict[str, type[UnaryExpr]] = {
    "Not": Not,
    "IsNull": IsNull,
    "IsNotNull": IsNotNull,
}


def parse_boolean_function(
    boolean_name: Any, inputs: list[Any]
) -> tuple[str, Any, tuple[Any, ...]]:
    """Extract IR constructor and child JSON for a Polars boolean `BooleanFunction` node."""
    if (
        isinstance(boolean_name, str)
        and boolean_name in BOOLEAN_UNARY_OPS
        and len(inputs) >= 1
    ):
        return ("Unary", BOOLEAN_UNARY_OPS[boolean_name], (inputs[0],))
    return UNSUPPORTED_RECORD


BOOLEAN_FUNCTION_PARSERS: dict[
    str, Callable[[Any, list[Any]], tuple[str, Any, tuple[Any, ...]]]
] = dict.fromkeys(BOOLEAN_UNARY_OPS, parse_boolean_function)
