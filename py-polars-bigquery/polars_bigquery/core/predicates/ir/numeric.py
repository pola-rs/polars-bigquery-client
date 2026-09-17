from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polars_bigquery.core.predicates.ir.base import (
    Expr,
    Literal,
    UnaryExpr,
    Unsupported,
)

INT_TYPES = frozenset(
    {
        "Int",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    }
)

FLOAT_TYPES = frozenset({"Float", "Float32", "Float64"})


@dataclass(frozen=True)
class IntLiteral(Literal):
    """IR node representing an integer literal."""

    value: int


@dataclass(frozen=True)
class FloatLiteral(Literal):
    """IR node representing a floating-point literal."""

    value: float


@dataclass(frozen=True)
class IsNan(UnaryExpr):
    """IR node representing an IS_NAN check on a numeric expression."""


@dataclass(frozen=True)
class IsNotNan(UnaryExpr):
    """IR node representing a NOT IS_NAN check on a numeric expression."""


@dataclass(frozen=True)
class IsInfinite(UnaryExpr):
    """IR node representing an IS_INF check on a numeric expression."""


@dataclass(frozen=True)
class IsFinite(UnaryExpr):
    """IR node representing a finite check (NOT IS_INF AND NOT IS_NAN) on a numeric expression."""


# Keys represent Polars Rust AST `BooleanFunction` enum variant names emitted under
# `{"Function": {"function": {"Boolean": "<key>"}}}` in serialized Polars JSON.
NUMERIC_UNARY_FUNCTIONS: dict[str, type[UnaryExpr]] = {
    "IsNan": IsNan,
    "IsNotNan": IsNotNan,
    "IsInfinite": IsInfinite,
    "IsFinite": IsFinite,
}


def parse_int_literal(value: Any) -> Expr:
    """Parse an integer value from Polars JSON into an IntLiteral or Unsupported."""
    try:
        return IntLiteral(value=int(value))
    except (TypeError, ValueError):
        return Unsupported()


def parse_float_literal(value: Any) -> Expr:
    """Parse a float value from Polars JSON into a FloatLiteral or Unsupported."""
    if value is None:
        return Unsupported()
    if isinstance(value, str):
        val_lower = value.strip().lower()
        if val_lower in ("nan", "+nan", "-nan"):
            return FloatLiteral(value=float("nan"))
        if val_lower in ("inf", "+inf", "infinity", "+infinity"):
            return FloatLiteral(value=float("inf"))
        if val_lower in ("-inf", "-infinity"):
            return FloatLiteral(value=float("-inf"))
    try:
        float_val = float(value)
    except (TypeError, ValueError):
        return Unsupported()
    return FloatLiteral(value=float_val)
