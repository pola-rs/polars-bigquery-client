from __future__ import annotations

from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    IsNotNull,
    IsNull,
    Not,
    Or,
)
from polars_bigquery.core.compiler.sql.base import (
    emit_node_sql,
    format_operator_sql,
    literal_ir_to_sql,
)


@literal_ir_to_sql.register
def format_bool_literal(node: BoolLiteral) -> str:
    """Format a BoolLiteral IR node as a BigQuery SQL boolean literal."""
    return "TRUE" if node.value else "FALSE"


@format_operator_sql.register
def format_not(_node: Not, operand: str) -> str:
    return f"(NOT {operand})"


@format_operator_sql.register
def format_is_null(_node: IsNull, operand: str) -> str:
    return f"({operand} IS NULL)"


@format_operator_sql.register
def format_is_not_null(_node: IsNotNull, operand: str) -> str:
    return f"({operand} IS NOT NULL)"


@emit_node_sql.register
def emit_and_sql(
    _node: And, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    (left_sql, left_exact), (right_sql, right_exact) = (
        child_results[0],
        child_results[1],
    )
    if left_sql is None and right_sql is None:
        return None, False
    if left_sql is None:
        return right_sql, False
    if right_sql is None:
        return left_sql, False
    return f"({left_sql} AND {right_sql})", (left_exact and right_exact)


@emit_node_sql.register
def emit_or_sql(
    _node: Or, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    (left_sql, left_exact), (right_sql, right_exact) = (
        child_results[0],
        child_results[1],
    )
    if left_sql is None or right_sql is None:
        return None, False
    return f"({left_sql} OR {right_sql})", (left_exact and right_exact)
