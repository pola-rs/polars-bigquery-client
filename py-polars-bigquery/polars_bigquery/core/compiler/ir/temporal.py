from __future__ import annotations

import dataclasses
import enum

from polars_bigquery.core.compiler.ir.base import Literal


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
