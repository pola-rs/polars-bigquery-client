from __future__ import annotations

import math

from polars_bigquery.core.compiler.ir.numeric import (
    FloatLiteral,
    IntLiteral,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
)
from polars_bigquery.core.compiler.sql.base import (
    format_operator_sql,
    literal_ir_to_sql,
)


@literal_ir_to_sql.register
def format_int_literal(node: IntLiteral) -> str:
    """Format an IntLiteral IR node as a BigQuery SQL integer literal."""
    return str(node.value)


@literal_ir_to_sql.register
def format_float_literal(node: FloatLiteral) -> str:
    """Format a FloatLiteral IR node as a BigQuery SQL FLOAT64 literal."""
    if math.isnan(node.value):
        return "CAST('nan' AS FLOAT64)"
    if math.isinf(node.value):
        return "CAST('-inf' AS FLOAT64)" if node.value < 0 else "CAST('inf' AS FLOAT64)"
    return repr(node.value)


@format_operator_sql.register
def format_is_nan(_node: IsNan, operand: str) -> str:
    return f"IS_NAN({operand})"


@format_operator_sql.register
def format_is_not_nan(_node: IsNotNan, operand: str) -> str:
    return f"(NOT IS_NAN({operand}))"


@format_operator_sql.register
def format_is_infinite(_node: IsInfinite, operand: str) -> str:
    return f"IS_INF({operand})"


@format_operator_sql.register
def format_is_finite(_node: IsFinite, operand: str) -> str:
    return f"(NOT IS_INF({operand}) AND NOT IS_NAN({operand}))"
