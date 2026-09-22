"""Functions for emitting SQL from an Expr IR tree."""

from __future__ import annotations

import collections

from polars_bigquery.core.compiler.ir import Expr
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
    nodes: list[Expr] = [root]
    children_map: list[tuple[int, ...]] = [()]
    queue: collections.deque[int] = collections.deque([0])

    while queue:
        curr_idx = queue.popleft()
        node = nodes[curr_idx]
        node_children = node.children()
        if node_children:
            child_indices: list[int] = []
            for child in node_children:
                c_idx = len(nodes)
                nodes.append(child)
                children_map.append(())
                queue.append(c_idx)
                child_indices.append(c_idx)
            children_map[curr_idx] = tuple(child_indices)

    results: list[tuple[str | None, bool]] = [(None, False)] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        node = nodes[idx]
        child_results = [results[c_idx] for c_idx in children_map[idx]]
        results[idx] = emit_node_sql(node, child_results)

    return results[0]


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
