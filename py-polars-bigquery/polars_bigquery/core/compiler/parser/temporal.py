from __future__ import annotations

from typing import Any

from polars_bigquery.core.compiler.ir.base import Expr, Unsupported
from polars_bigquery.core.compiler.ir.temporal import (
    DateLiteral,
    TimestampLiteral,
    TimestampUnit,
)
from polars_bigquery.core.compiler.parser.base import register_parser

# BigQuery DATE / TIMESTAMP bounds: 0001-01-01T00:00:00Z to 9999-12-31T23:59:59.999999Z
BQ_MIN_DATE_DAYS = -719_162
BQ_MAX_DATE_DAYS = 2_932_896
BQ_MIN_TIMESTAMP_MICROS = -62_135_596_800_000_000
BQ_MAX_TIMESTAMP_MICROS = 253_402_300_799_999_999


@register_parser("$.Literal..Date")
def parse_date_literal(value: Any) -> Expr:
    """Parse a Date value from Polars JSON into a DateLiteral or Unsupported."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return Unsupported()
    try:
        days = int(value)
    except (TypeError, ValueError):
        return Unsupported()
    if not (BQ_MIN_DATE_DAYS <= days <= BQ_MAX_DATE_DAYS):
        return Unsupported()
    return DateLiteral(days=days)


def parse_ticks_and_unit(ticks_raw: Any, units_raw: Any) -> tuple[int, TimestampUnit]:
    """Parse ticks and unit from Polars JSON, converting exact nanoseconds to microseconds."""
    if isinstance(ticks_raw, bool) or not isinstance(ticks_raw, (int, str)):
        raise TypeError("Invalid timestamp tick count type")
    ticks = int(ticks_raw)
    if units_raw == "Microseconds":
        if not (BQ_MIN_TIMESTAMP_MICROS <= ticks <= BQ_MAX_TIMESTAMP_MICROS):
            raise ValueError("Timestamp microseconds out of BigQuery bounds")
        return ticks, TimestampUnit.MICROSECONDS
    if units_raw == "Milliseconds":
        if not (BQ_MIN_TIMESTAMP_MICROS <= ticks * 1000 <= BQ_MAX_TIMESTAMP_MICROS):
            raise ValueError("Timestamp milliseconds out of BigQuery bounds")
        return ticks, TimestampUnit.MILLISECONDS
    if units_raw == "Nanoseconds" and ticks % 1000 == 0:
        micros = ticks // 1000
        if not (BQ_MIN_TIMESTAMP_MICROS <= micros <= BQ_MAX_TIMESTAMP_MICROS):
            raise ValueError("Timestamp nanoseconds out of BigQuery bounds")
        return micros, TimestampUnit.MICROSECONDS
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
