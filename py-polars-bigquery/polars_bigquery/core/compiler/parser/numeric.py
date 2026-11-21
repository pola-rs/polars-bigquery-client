from __future__ import annotations

import math
import struct
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

BQ_INT64_MIN = -(1 << 63)
BQ_INT64_MAX = (1 << 63) - 1


@register_parser("$.Function.function.Boolean.IsNan")
def parse_is_nan(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsNan` BooleanFunction node into an IR `IsNan` record."""
    return parse_unary_function(expr_json, IsNan, func_spec=spec)


@register_parser("$.Function.function.Boolean.IsNotNan")
def parse_is_not_nan(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsNotNan` BooleanFunction node into an IR `IsNotNan` record."""
    return parse_unary_function(expr_json, IsNotNan, func_spec=spec)


@register_parser("$.Function.function.Boolean.IsInfinite")
def parse_is_infinite(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsInfinite` BooleanFunction node into an IR `IsInfinite` record."""
    return parse_unary_function(expr_json, IsInfinite, func_spec=spec)


@register_parser("$.Function.function.Boolean.IsFinite")
def parse_is_finite(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsFinite` BooleanFunction node into an IR `IsFinite` record."""
    return parse_unary_function(expr_json, IsFinite, func_spec=spec)


@register_parser(*(f"$.Literal..{int_type}" for int_type in sorted(INT_TYPES)))
def parse_int_literal(value: Any) -> Expr:
    """Parse an integer value from Polars JSON into an IntLiteral or Unsupported."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return Unsupported()
    try:
        int_val = int(value)
    except (TypeError, ValueError):
        return Unsupported()
    if not (BQ_INT64_MIN <= int_val <= BQ_INT64_MAX):
        return Unsupported()
    return IntLiteral(value=int_val)


def _parse_float_value(value: Any) -> float | None:
    """Parse a JSON float payload into a Python float, or None when invalid/overflow."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if isinstance(value, str):
        val_lower = value.strip().lower()
        if val_lower in ("nan", "+nan", "-nan"):
            return float("nan")
        if val_lower in ("inf", "+inf", "infinity", "+infinity"):
            return float("inf")
        if val_lower in ("-inf", "-infinity"):
            return float("-inf")
    try:
        float_val = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isinf(float_val) and not (isinstance(value, float) and math.isinf(value)):
        return None
    return float_val


@register_parser("$.Literal..Float", "$.Literal..Float64")
def parse_float_literal(value: Any) -> Expr:
    """Parse a Float/Float64 value from Polars JSON into a FloatLiteral or Unsupported."""
    float_val = _parse_float_value(value)
    if float_val is None:
        return Unsupported()
    return FloatLiteral(value=float_val)


@register_parser("$.Literal..Float32")
def parse_float32_literal(value: Any) -> Expr:
    """Parse a Float32 value from Polars JSON preserving IEEE-754 single-precision rounding.

    Polars serializes `Float32` literals as their shortest decimal form (e.g.
    `0.1` instead of `0.10000000149011612`). Round-tripping finite values through
    IEEE-754 binary32 keeps BigQuery `FLOAT64` comparisons aligned with Polars'
    `f32 -> f64` promotion so equality pushdown does not drop matching rows.
    """
    float_val = _parse_float_value(value)
    if float_val is None:
        return Unsupported()
    if math.isnan(float_val) or math.isinf(float_val):
        return FloatLiteral(value=float_val)
    try:
        widened = struct.unpack("<f", struct.pack("<f", float_val))[0]
    except OverflowError:
        return Unsupported()
    if math.isinf(widened):
        return Unsupported()
    return FloatLiteral(value=widened)
