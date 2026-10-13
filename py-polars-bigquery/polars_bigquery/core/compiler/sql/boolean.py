from __future__ import annotations

from polars_bigquery.core.compiler.ir.base import NullLiteral
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    Coalesce,
    FillNull,
    IsNotNull,
    IsNull,
    Not,
    Or,
    Ternary,
    When,
)
from polars_bigquery.core.compiler.ir.list_ import ListLiteral
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


@format_operator_sql.register
def format_coalesce(_node: Coalesce, *operands: str) -> str | None:
    if not operands:
        return None
    return f"COALESCE({', '.join(operands)})"


@format_operator_sql.register
def format_fill_null(_node: FillNull, left: str, right: str) -> str:
    return f"IFNULL({left}, {right})"


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


@emit_node_sql.register
def emit_ternary_sql(
    node: Ternary, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    """Emit a BigQuery `IF(predicate, truthy, falsy)` SQL expression for a `Ternary` node.

    Requires all three operands to be supported and exact, none of the operands
    to be a `ListLiteral`, `predicate` not to be a `NullLiteral`, and at least
    one of `truthy` or `falsy` to be non-`NullLiteral`.
    """
    if len(child_results) != 3:
        return None, False
    (pred_sql, pred_exact), (truthy_sql, truthy_exact), (falsy_sql, falsy_exact) = (
        child_results[0],
        child_results[1],
        child_results[2],
    )
    if (
        pred_sql is None
        or truthy_sql is None
        or falsy_sql is None
        or not (pred_exact and truthy_exact and falsy_exact)
    ):
        return None, False
    if isinstance(node.predicate, (NullLiteral, ListLiteral)):
        return None, False
    if isinstance(node.truthy, ListLiteral) or isinstance(node.falsy, ListLiteral):
        return None, False
    if isinstance(node.truthy, NullLiteral) and isinstance(node.falsy, NullLiteral):
        return None, False

    return f"IF({pred_sql}, {truthy_sql}, {falsy_sql})", True


@emit_node_sql.register
def emit_when_sql(
    node: When, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    """Emit a BigQuery `CASE WHEN ... THEN ... [ELSE ...] END` SQL expression for a `When` node.

    Expects `node.operands` as `(predicate_1, truthy_1, ..., predicate_n, truthy_n, falsy)`.
    Omits the trailing `ELSE` clause when `falsy` is a `NullLiteral`, and degrades to
    `(None, False)` if any operand is unsupported, inexact, or a `ListLiteral`, if any
    predicate is a `NullLiteral`, or if all output branches are `NullLiteral`.
    """
    # Operands must be (pred_1, truthy_1, ..., pred_n, truthy_n, falsy).
    if (
        len(child_results) < 3
        or len(child_results) % 2 == 0
        or len(node.operands) != len(child_results)
    ):
        return None, False
    if any(sql is None or not is_exact for sql, is_exact in child_results):
        return None, False
    if any(isinstance(op, ListLiteral) for op in node.operands):
        return None, False
    # Even indices before the last element are WHEN predicates.
    if any(isinstance(pred, NullLiteral) for pred in node.operands[:-1:2]):
        return None, False
    # Odd indices (THEN branches) + final element (ELSE branch) cannot all be NULL.
    if all(
        isinstance(branch, NullLiteral)
        for branch in (*node.operands[1::2], node.operands[-1])
    ):
        return None, False

    when_clauses = " ".join(
        f"WHEN {child_results[i][0]} THEN {child_results[i + 1][0]}"
        for i in range(0, len(child_results) - 1, 2)
    )
    if isinstance(node.operands[-1], NullLiteral):
        return f"CASE {when_clauses} END", True

    falsy_sql = child_results[-1][0]
    return f"CASE {when_clauses} ELSE {falsy_sql} END", True
