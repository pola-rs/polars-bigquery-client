"""Functions for emitting SQL from an Expr IR tree."""

from __future__ import annotations

from polars_bigquery.core.compiler._traversal import flatten_expr_bfs
from polars_bigquery.core.compiler.ir import And, Expr, NullLiteral
from polars_bigquery.core.compiler.sql import (
    base,
    boolean,
    comparison,
    list_,
    numeric,
    string,
    temporal,
)
from polars_bigquery.core.compiler.sql.base import (
    column_to_sql_identifier,
    emit_node_sql,
    format_operator_sql,
    literal_ir_to_sql,
)
from polars_bigquery.core.compiler.sql.string import escape_sql_string

_column_to_sql_identifier = column_to_sql_identifier
_emit_node_sql = emit_node_sql
_escape_sql_string = escape_sql_string
_format_operator_sql = format_operator_sql
_is_allowed_unicode_category = base._is_allowed_unicode_category
_is_valid_column_character = base._is_valid_column_character
_literal_ir_to_sql = literal_ir_to_sql


def _compile_ir_to_sql(root: Expr) -> tuple[str | None, bool]:
    """Compile an Expr IR tree into a BigQuery SQL string and an exactness boolean.

    Performs a breadth-first walk using a queue to traverse the IR tree top-down,
    then evaluates SQL strings bottom-up in reverse BFS order to avoid stack size
    limitations.
    """
    nodes, children_map = flatten_expr_bfs(root)

    results: list[tuple[str | None, bool]] = [(None, False)] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        node = nodes[idx]
        child_results = [results[c_idx] for c_idx in children_map[idx]]
        results[idx] = emit_node_sql(node, child_results)

    return results[0]


def compile_ir_to_sql(root: Expr) -> tuple[str | None, bool]:
    """Compile an Expr IR tree into `(sql, is_exact)`."""
    return _compile_ir_to_sql(root)


def compile_ir_conjuncts_to_sql(root: Expr) -> tuple[tuple[str, ...], bool]:
    """Flatten top-level `And` nodes left-to-right and compile each conjunct.

    Uses an explicit stack to handle arbitrarily deep `And` trees without
    hitting Python's recursion limit. Trivial `"TRUE"` conjuncts are omitted
    while preserving `is_exact=True`. Bare `NullLiteral` conjuncts are omitted
    with `is_exact=False`. If any top-level branch compiles to `(None, _)` or
    `(_, False)`, `is_exact` is returned as `False`.
    """
    sql_conjuncts: list[str] = []
    all_exact = True
    stack: list[Expr] = [root]

    while stack:
        node = stack.pop()
        if isinstance(node, And):
            # Push right first so left is popped and visited first.
            stack.append(node.right)
            stack.append(node.left)
            continue

        if isinstance(node, NullLiteral):
            all_exact = False
            continue

        sql_str, is_exact = _compile_ir_to_sql(node)
        if sql_str is None:
            all_exact = False
        else:
            if not is_exact:
                all_exact = False
            if sql_str != "TRUE":
                sql_conjuncts.append(sql_str)

    return tuple(sql_conjuncts), all_exact


def ir_to_sql(root: Expr) -> str | None:
    """Compile an Expr IR tree into a BigQuery SQL row restriction string.

    Returns None if the root expression cannot be safely converted.
    """
    sql, _is_exact = _compile_ir_to_sql(root)
    return sql


__all__ = [
    "base",
    "boolean",
    "column_to_sql_identifier",
    "comparison",
    "compile_ir_conjuncts_to_sql",
    "compile_ir_to_sql",
    "emit_node_sql",
    "escape_sql_string",
    "format_operator_sql",
    "ir_to_sql",
    "list_",
    "literal_ir_to_sql",
    "numeric",
    "string",
    "temporal",
]
