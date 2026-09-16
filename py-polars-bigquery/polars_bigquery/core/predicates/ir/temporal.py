from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from polars_bigquery.core.predicates.ir.base import Expr, Literal, Unsupported


class TimestampUnit(enum.Enum):
    """Supported timestamp units in the predicate IR."""

    MICROSECONDS = "Microseconds"
    MILLISECONDS = "Milliseconds"


@dataclass(frozen=True)
class DateLiteral(Literal):
    """IR node representing a date literal (days since UNIX epoch)."""

    days: int


@dataclass(frozen=True)
class TimestampLiteral(Literal):
    """IR node representing a timestamp literal."""

    ticks: int
    unit: TimestampUnit


def parse_date_literal(value: Any) -> Expr:
    """Parse a Date value from Polars JSON into a DateLiteral or Unsupported."""
    try:
        days = int(value)
    except (TypeError, ValueError):
        return Unsupported()
    return DateLiteral(days=days)


def parse_datetime_literal(value: Any) -> Expr:
    """Parse a DateTime value from Polars JSON into a TimestampLiteral or Unsupported."""
    try:
        ticks = int(value[0])
        units = value[1]
    except (IndexError, KeyError, TypeError, ValueError):
        return Unsupported()
    if units == "Microseconds":
        return TimestampLiteral(ticks=ticks, unit=TimestampUnit.MICROSECONDS)
    if units == "Milliseconds":
        return TimestampLiteral(ticks=ticks, unit=TimestampUnit.MILLISECONDS)
    if units == "Nanoseconds" and ticks % 1000 == 0:
        return TimestampLiteral(ticks=ticks // 1000, unit=TimestampUnit.MICROSECONDS)
    return Unsupported()
