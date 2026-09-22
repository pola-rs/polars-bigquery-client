from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from polars_bigquery.core.predicates.ir.base import (
    BinaryExpr,
    Expr,
    UnaryExpr,
)
from polars_bigquery.core.predicates.ir.string import (
    Contains,
    EndsWith,
    Lowercase,
    StartsWith,
    StringLiteral,
    Uppercase,
)
from polars_bigquery.core.predicates.parser.base import UNSUPPORTED_RECORD

STRING_TYPES = frozenset({"String", "StringOwned"})

# Keys represent Polars Rust AST `StringFunction` enum variant names emitted under
# `{"Function": {"function": {"StringExpr": "<key>"}}}` in serialized Polars JSON.
STRING_UNARY_OPS: dict[str, type[UnaryExpr]] = {
    "Uppercase": Uppercase,
    "Lowercase": Lowercase,
}

STRING_BINARY_OPS: dict[str, type[BinaryExpr]] = {
    "StartsWith": StartsWith,
    "EndsWith": EndsWith,
}


def parse_string_literal(value: Any) -> StringLiteral:
    """Parse a string value from Polars JSON into a StringLiteral."""
    return StringLiteral(value=str(value))


# Keys represent Polars Rust AST `LiteralValue` / `AnyValue` string variant names
# emitted inside `{"Literal": ...}` in serialized Polars JSON.
STRING_LITERAL_PARSERS: dict[str, Callable[[Any], Expr]] = dict.fromkeys(
    STRING_TYPES, parse_string_literal
)


def parse_string_function(
    string_name: Any, inputs: list[Any]
) -> tuple[str, Any, tuple[Any, ...]]:
    """Extract IR constructor and child JSONs for a Polars `StringFunction` node."""
    if (
        isinstance(string_name, str)
        and string_name in STRING_UNARY_OPS
        and len(inputs) == 1
    ):
        return ("Unary", STRING_UNARY_OPS[string_name], (inputs[0],))
    if (
        isinstance(string_name, str)
        and string_name in STRING_BINARY_OPS
        and len(inputs) == 2
    ):
        return ("Binary", STRING_BINARY_OPS[string_name], (inputs[0], inputs[1]))
    if (
        isinstance(string_name, dict)
        and len(string_name) == 1
        and "Contains" in string_name
        and len(inputs) == 2
    ):
        contains_opts = string_name["Contains"]
        if isinstance(contains_opts, dict):
            literal = bool(contains_opts.get("literal", False))
            return (
                "Binary",
                functools.partial(Contains, literal=literal),
                (inputs[0], inputs[1]),
            )
    return UNSUPPORTED_RECORD
