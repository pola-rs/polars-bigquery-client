from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable
from typing import Any

from polars_bigquery.core.predicates.ir.base import Expr, Literal, Unsupported


class TimestampUnit(enum.Enum):
    """Supported timestamp units in the predicate IR."""

    MICROSECONDS = "Microseconds"
    MILLISECONDS = "Milliseconds"


@dataclasses.dataclass(frozen=True)
class DateLiteral(Literal):
    """IR node representing a date literal (days since UNIX epoch)."""

    days: int


@dataclasses.dataclass(frozen=True)
class TimestampLiteral(Literal):
    """IR node representing a timestamp or timezone-naive datetime literal."""

    ticks: int
    unit: TimestampUnit
    tz: str | None = "UTC"


def parse_date_literal(value: Any) -> Expr:
    """Parse a Date value from Polars JSON into a DateLiteral or Unsupported."""
    try:
        days = int(value)
    except (TypeError, ValueError):
        return Unsupported()
    return DateLiteral(days=days)


def parse_datetime_literal(value: Any) -> Expr:
    """Parse a DateTime/Datetime value from Polars JSON into a TimestampLiteral or Unsupported."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return Unsupported()
    try:
        ticks = int(value[0])
        units = value[1]
    except (IndexError, KeyError, TypeError, ValueError):
        return Unsupported()

    tz: str | None = "UTC"
    if len(value) >= 3:
        tz_raw = value[2]
        if tz_raw is None:
            tz = None
        elif isinstance(tz_raw, str):
            tz = tz_raw
        elif isinstance(tz_raw, dict) and isinstance(tz_raw.get("inner"), str):
            tz = tz_raw["inner"]
        else:
            return Unsupported()

    if units == "Microseconds":
        return TimestampLiteral(ticks=ticks, unit=TimestampUnit.MICROSECONDS, tz=tz)
    if units == "Milliseconds":
        return TimestampLiteral(ticks=ticks, unit=TimestampUnit.MILLISECONDS, tz=tz)
    if units == "Nanoseconds" and ticks % 1000 == 0:
        return TimestampLiteral(
            ticks=ticks // 1000, unit=TimestampUnit.MICROSECONDS, tz=tz
        )
    return Unsupported()


# Keys represent Polars Rust AST `LiteralValue` / `AnyValue` temporal variant names
# emitted inside `{"Literal": ...}` in serialized Polars JSON.
TEMPORAL_LITERAL_PARSERS: dict[str, Callable[[Any], Expr]] = {
    "Date": parse_date_literal,
    "Datetime": parse_datetime_literal,
    "DateTime": parse_datetime_literal,
}
