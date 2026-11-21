from __future__ import annotations

from polars_bigquery.core.compiler.ir.temporal import (
    DateLiteral,
    TimeLiteral,
    TimestampLiteral,
    TimestampUnit,
)
from polars_bigquery.core.compiler.sql.base import literal_ir_to_sql

_MICROS_PER_SECOND = 1_000_000
_MICROS_PER_MINUTE = 60 * _MICROS_PER_SECOND
_MICROS_PER_HOUR = 60 * _MICROS_PER_MINUTE
_MICROS_PER_DAY = 24 * _MICROS_PER_HOUR


@literal_ir_to_sql.register
def format_date_literal(node: DateLiteral) -> str:
    """Format a DateLiteral IR node as a BigQuery SQL DATE expression."""
    return f"DATE(TIMESTAMP_SECONDS({node.days} * 86400))"


@literal_ir_to_sql.register
def format_time_literal(node: TimeLiteral) -> str:
    """Format a TimeLiteral IR node as a BigQuery SQL TIME literal."""
    micros = node.micros
    if (
        not isinstance(micros, int)
        or isinstance(micros, bool)
        or not (0 <= micros < _MICROS_PER_DAY)
    ):
        msg = (
            f"Invalid TimeLiteral micros {micros!r}: expected int in "
            f"[0, {_MICROS_PER_DAY})."
        )
        raise ValueError(msg)
    hours, rem = divmod(micros, _MICROS_PER_HOUR)
    minutes, rem = divmod(rem, _MICROS_PER_MINUTE)
    seconds, subsec_micros = divmod(rem, _MICROS_PER_SECOND)
    if subsec_micros:
        return f"TIME '{hours:02d}:{minutes:02d}:{seconds:02d}.{subsec_micros:06d}'"
    return f"TIME '{hours:02d}:{minutes:02d}:{seconds:02d}'"


@literal_ir_to_sql.register
def format_timestamp_literal(node: TimestampLiteral) -> str:
    """Format a TimestampLiteral IR node as a BigQuery SQL TIMESTAMP or DATETIME expression."""
    if node.unit == TimestampUnit.MICROSECONDS:
        ts_sql = f"TIMESTAMP_MICROS({node.ticks})"
    elif node.unit == TimestampUnit.MILLISECONDS:
        ts_sql = f"TIMESTAMP_MILLIS({node.ticks})"
    else:
        msg = f"Unhandled timestamp unit: {node.unit!r}"
        raise TypeError(msg)
    return f"DATETIME({ts_sql})" if node.tz is None else ts_sql
