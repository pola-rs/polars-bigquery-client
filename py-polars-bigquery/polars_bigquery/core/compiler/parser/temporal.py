from __future__ import annotations

from typing import Any

from polars_bigquery.core.compiler.ir.base import Expr, Unsupported
from polars_bigquery.core.compiler.ir.temporal import (
    DateLiteral,
    TimestampLiteral,
    TimestampUnit,
)
from polars_bigquery.core.compiler.parser.base import register_parser


@register_parser("$.Literal..Date")
def parse_date_literal(value: Any) -> Expr:
    """Parse a Date value from Polars JSON into a DateLiteral or Unsupported."""
    if isinstance(value, bool):
        return Unsupported()
    try:
        days = int(value)
    except (TypeError, ValueError):
        return Unsupported()
    return DateLiteral(days=days)


def parse_ticks_and_unit(ticks_raw: Any, units_raw: Any) -> tuple[int, TimestampUnit]:
    """Parse ticks and unit from Polars JSON, converting exact nanoseconds to microseconds."""
    if isinstance(ticks_raw, bool):
        raise TypeError("Boolean is not a valid timestamp tick count")
    ticks = int(ticks_raw)
    if units_raw == "Microseconds":
        return ticks, TimestampUnit.MICROSECONDS
    if units_raw == "Milliseconds":
        return ticks, TimestampUnit.MILLISECONDS
    if units_raw == "Nanoseconds" and ticks % 1000 == 0:
        return ticks // 1000, TimestampUnit.MICROSECONDS
    raise ValueError(f"Unsupported datetime unit or precision: {units_raw!r}")


def parse_timezone(tz_raw: Any) -> str | None:
    """Parse a timezone representation from Polars JSON."""
    if tz_raw is None or isinstance(tz_raw, str):
        return tz_raw
    if isinstance(tz_raw, dict) and isinstance(tz_raw.get("inner"), str):
        return tz_raw["inner"]
    raise ValueError(f"Unsupported timezone representation: {tz_raw!r}")


@register_parser("$.Literal..Datetime", "$.Literal..DateTime")
def parse_datetime_literal(value: Any) -> Expr:
    """Parse a DateTime/Datetime value from Polars JSON into a TimestampLiteral or Unsupported."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return Unsupported()
    try:
        ticks, unit = parse_ticks_and_unit(value[0], value[1])
        tz = parse_timezone(value[2] if len(value) >= 3 else "UTC")
    except (TypeError, ValueError):
        return Unsupported()
    return TimestampLiteral(ticks=ticks, unit=unit, tz=tz)
