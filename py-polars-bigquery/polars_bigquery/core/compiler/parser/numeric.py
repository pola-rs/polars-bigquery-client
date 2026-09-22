from __future__ import annotations

from collections.abc import Callable
from typing import Any

from polars_bigquery.core.compiler.ir.base import (
    Expr,
    UnaryExpr,
    Unsupported,
)
from polars_bigquery.core.compiler.ir.numeric import (
    FloatLiteral,
    IntLiteral,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
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

# Keys represent Polars Rust AST `BooleanFunction` enum variant names emitted under
# `{"Function": {"function": {"Boolean": "<key>"}}}` in serialized Polars JSON.
NUMERIC_UNARY_OPS: dict[str, type[UnaryExpr]] = {
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


# Keys represent Polars Rust AST `LiteralValue` / `AnyValue` numeric variant names
# emitted inside `{"Literal": ...}` in serialized Polars JSON.
NUMERIC_LITERAL_PARSERS: dict[str, Callable[[Any], Expr]] = {
    **dict.fromkeys(INT_TYPES, parse_int_literal),
    **dict.fromkeys(FLOAT_TYPES, parse_float_literal),
}
