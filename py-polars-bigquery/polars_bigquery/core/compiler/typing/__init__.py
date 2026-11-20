"""Pre-rewrite schema-aware typing and lowering pass for the predicate compiler."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from polars_bigquery.core.compiler._traversal import flatten_expr_bfs
from polars_bigquery.core.compiler.ir.base import Expr, NullLiteral, Unsupported
from polars_bigquery.core.compiler.ir.list_ import ListLiteral
from polars_bigquery.core.compiler.parser import MAX_AST_NODES
from polars_bigquery.core.compiler.typing import (
    base,
    boolean,
    comparison,
    list_,
    literals,
    numeric,
    string,
)
from polars_bigquery.core.compiler.typing.base import (
    _UNSUPPORTED_TYPED_NODE,
    MAX_DUPLICATED_OPERAND_NODES,
    _type_node,
    _TypedNode,
    _TypingContext,
)
from polars_bigquery.core.types import BQType

logger = logging.getLogger(__name__)


def resolve_and_lower_types(
    root: Expr,
    schema: pl.Schema,
    bq_types: Mapping[str, BQType] | None = None,
) -> Expr:
    """Resolve column/expression types and lower schema-dependent IR nodes iteratively.

    Performs a two-pass iterative traversal (zero Python recursion):
    1. Pass 1 (Top-down BFS indexing): Flattens `root` into breadth-first order and
       computes total input AST node weight (counting `ListLiteral` as `1 + len(values)`).
    2. Pass 2 (Bottom-up reverse-BFS evaluation): Evaluates types, enforces operator
       compatibility, lowers `FillNullWithStrategy`, `INT64` float predicates, and
       `FLOAT64` `NaN` comparisons within `MAX_AST_NODES` (refunding wasted budget when
       a lowered subtree degrades to `Unsupported`), and validates the root type.
    """
    nodes, children_map = flatten_expr_bfs(root)
    total_input_weight = sum(
        1 + len(node.values) if isinstance(node, ListLiteral) else 1 for node in nodes
    )

    ctx = _TypingContext(
        schema=schema,
        bq_types=bq_types,
        remaining_budget=max(0, MAX_AST_NODES - total_input_weight),
    )

    typed_nodes: list[_TypedNode] = [_UNSUPPORTED_TYPED_NODE] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        child_typed = tuple(typed_nodes[c_idx] for c_idx in children_map[idx])
        typed_nodes[idx] = _type_node(nodes[idx], child_typed, ctx)

    root_typed = typed_nodes[0]
    if (
        isinstance(root_typed.node, NullLiteral)
        or root_typed.is_all_null
        or (root_typed.bq_type is not None and root_typed.bq_type is not BQType.BOOL)
    ):
        logger.debug(
            "Root predicate has non-BOOL or NULL type %r; degrading to Unsupported",
            root_typed.bq_type,
        )
        return Unsupported()

    return root_typed.node


__all__ = [
    "MAX_AST_NODES",
    "MAX_DUPLICATED_OPERAND_NODES",
    "base",
    "boolean",
    "comparison",
    "list_",
    "literals",
    "numeric",
    "resolve_and_lower_types",
    "string",
]
