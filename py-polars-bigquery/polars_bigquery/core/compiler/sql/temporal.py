from __future__ import annotations

from polars_bigquery.core.compiler.ir.temporal import (
    DateLiteral,
    TimestampLiteral,
    TimestampUnit,
)
from polars_bigquery.core.compiler.sql.base import literal_ir_to_sql


@literal_ir_to_sql.register
def format_date_literal(node: DateLiteral) -> str:
    """Format a DateLiteral IR node as a BigQuery SQL DATE expression."""
    return f"DATE(TIMESTAMP_SECONDS({node.days} * 86400))"


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
