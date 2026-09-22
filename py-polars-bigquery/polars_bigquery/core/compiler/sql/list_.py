from __future__ import annotations

import math

from polars_bigquery.core.compiler.ir.base import NullLiteral
from polars_bigquery.core.compiler.ir.list_ import IsIn, ListLiteral
from polars_bigquery.core.compiler.ir.numeric import FloatLiteral
from polars_bigquery.core.compiler.sql.base import (
    emit_node_sql,
    format_operator_sql,
    literal_ir_to_sql,
)


@literal_ir_to_sql.register
def format_list_literal(node: ListLiteral) -> str:
    """Format a ListLiteral IR node as a parenthesized SQL tuple."""
    items_sql = ", ".join(literal_ir_to_sql(v) for v in node.values)
    return f"({items_sql})"


@format_operator_sql.register
def format_is_in(_node: IsIn, left: str, right: str) -> str:
    """Format an IsIn IR node as a BigQuery SQL IN expression."""
    return f"({left} IN {right})"


@emit_node_sql.register
def emit_list_literal_sql(
    node: ListLiteral, _child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    """Emit the BigQuery SQL tuple string for a ListLiteral node if all items are valid."""
    if not node.values or any(
        isinstance(v, (NullLiteral, ListLiteral))
        or (isinstance(v, FloatLiteral) and math.isnan(v.value))
        for v in node.values
    ):
        return None, False
    return format_list_literal(node), True


@emit_node_sql.register
def emit_is_in_sql(
    node: IsIn, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    """Emit the BigQuery SQL string for an IsIn node if the RHS is a valid ListLiteral."""
    if any(sql is None or not is_exact for sql, is_exact in child_results):
        return None, False
    if isinstance(node.left, (NullLiteral, ListLiteral)) or not isinstance(
        node.right, ListLiteral
    ):
        return None, False
    left_sql, _ = child_results[0]
    right_sql, _ = child_results[1]
    if left_sql is None or right_sql is None:
        return None, False
    return format_is_in(node, left_sql, right_sql), True
