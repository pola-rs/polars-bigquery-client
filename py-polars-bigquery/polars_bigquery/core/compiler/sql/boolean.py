from __future__ import annotations

from polars_bigquery.core.compiler.ir.base import NullLiteral
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    Coalesce,
    IsNotNull,
    IsNull,
    Not,
    Or,
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
def emit_when_sql(
    node: When, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
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

    if isinstance(node.falsy, NullLiteral):
        return f"CASE WHEN {pred_sql} THEN {truthy_sql} END", True

    if isinstance(node.falsy, When):
        # Polars represents chained `.when(p1).then(t1).when(p2).then(t2)` as
        # nested `When` nodes where the inner `When` is the `falsy` child of
        # the outer `When`. Because SQL emission is bottom-up, `falsy_sql` is
        # already compiled as `"CASE WHEN <p2> THEN <t2> ... END"`. Strip the
        # outer `"CASE "` and `" END"` so the chain flattens into a single
        # `CASE WHEN <p1> THEN <t1> WHEN <p2> THEN <t2> ... END` expression
        # instead of emitting nested `ELSE CASE WHEN ... END END`.
        if falsy_sql.startswith("CASE WHEN ") and falsy_sql.endswith(" END"):
            inner_clauses = falsy_sql.removeprefix("CASE ").removesuffix(" END")
            return f"CASE WHEN {pred_sql} THEN {truthy_sql} {inner_clauses} END", True
        return None, False

    return f"CASE WHEN {pred_sql} THEN {truthy_sql} ELSE {falsy_sql} END", True
