from __future__ import annotations

from typing import Any

from polars_bigquery.core.compiler.ir.base import (
    Expr,
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
from polars_bigquery.core.compiler.parser.base import (
    ParseRecord,
    parse_unary_function,
    register_parser,
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


@register_parser("$.Function.function.Boolean.IsNan")
def parse_is_nan(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsNan` BooleanFunction node into an IR `IsNan` record."""
    return parse_unary_function(expr_json, IsNan)


@register_parser("$.Function.function.Boolean.IsNotNan")
def parse_is_not_nan(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsNotNan` BooleanFunction node into an IR `IsNotNan` record."""
    return parse_unary_function(expr_json, IsNotNan)


@register_parser("$.Function.function.Boolean.IsInfinite")
def parse_is_infinite(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsInfinite` BooleanFunction node into an IR `IsInfinite` record."""
    return parse_unary_function(expr_json, IsInfinite)


@register_parser("$.Function.function.Boolean.IsFinite")
def parse_is_finite(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsFinite` BooleanFunction node into an IR `IsFinite` record."""
    return parse_unary_function(expr_json, IsFinite)


@register_parser(*(f"$.Literal..{int_type}" for int_type in sorted(INT_TYPES)))
def parse_int_literal(value: Any) -> Expr:
    """Parse an integer value from Polars JSON into an IntLiteral or Unsupported."""
    if isinstance(value, bool):
        return Unsupported()
    try:
        return IntLiteral(value=int(value))
    except (TypeError, ValueError):
        return Unsupported()


@register_parser(*(f"$.Literal..{float_type}" for float_type in sorted(FLOAT_TYPES)))
def parse_float_literal(value: Any) -> Expr:
    """Parse a float value from Polars JSON into a FloatLiteral or Unsupported."""
    if value is None or isinstance(value, bool):
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
