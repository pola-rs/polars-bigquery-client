from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from polars_bigquery.core.predicates.ir.base import (
    BinaryExpr,
    Expr,
    Literal,
    UnaryExpr,
)


@dataclasses.dataclass(frozen=True)
class BoolLiteral(Literal):
    """IR node representing a boolean literal."""

    value: bool


@dataclasses.dataclass(frozen=True)
class And(BinaryExpr):
    """IR node representing logical conjunction (AND)."""


@dataclasses.dataclass(frozen=True)
class Or(BinaryExpr):
    """IR node representing logical disjunction (OR)."""


@dataclasses.dataclass(frozen=True)
class Not(UnaryExpr):
    """IR node representing logical negation (NOT)."""


@dataclasses.dataclass(frozen=True)
class IsNull(UnaryExpr):
    """IR node representing an IS NULL check."""


@dataclasses.dataclass(frozen=True)
class IsNotNull(UnaryExpr):
    """IR node representing an IS NOT NULL check."""


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
