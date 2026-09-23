"""Functions for parsing Polars Expr JSON into an IR Expr tree."""

from __future__ import annotations

import collections
from typing import Any

from polars_bigquery.core.compiler.ir import (
    Expr,
    ListLiteral,
    Unsupported,
)
from polars_bigquery.core.compiler.parser import (
    base,
    boolean,
    comparison,
    list_,
    numeric,
    string,
    temporal,
)
from polars_bigquery.core.compiler.parser.base import (
    PARSERS,
    UNSUPPORTED_RECORD,
    dispatch_parser,
    parse_json_path,
    register_parser,
)

MAX_AST_NODES = 10_000


def json_to_ir(expr_json: Any) -> Expr:
    """Compile Polars expression JSON into an immutable Expr IR tree.

    Performs a two-pass iterative compilation to avoid Python's recursion limit
    on deeply nested expression trees:
    1. Pass 1 (Top-down BFS traversal): Walks the JSON tree using a FIFO queue,
       dispatches each node against the registered JSON Path parsers, and records
       its kind, constructor (or leaf), and child indices.
    2. Pass 2 (Bottom-up IR construction): Iterates over the recorded nodes in
       reverse BFS order so every child `Expr` is instantiated before its parent.
    """
    # Pass 1: Top-down breadth-first discovery.
    # `raw_nodes` stores the raw JSON sub-expressions indexed by integer ID.
    # When node `curr_idx` is popped from `queue`, any child JSON sub-expressions
    # are appended to `raw_nodes` and `queue`. Because children are always appended
    # after their parent is popped, `parent_idx < child_idx` holds for every edge
    # in the expression tree, and `records[curr_idx]` stores the child indices.
    raw_nodes: list[Any] = [expr_json]
    records: list[tuple[str, Any, tuple[int, ...]]] = []
    queue: collections.deque[int] = collections.deque([0])
    ipc_leaf_nodes = 0

    while queue:
        curr_idx = queue.popleft()
        curr = raw_nodes[curr_idx]

        if len(raw_nodes) + ipc_leaf_nodes > MAX_AST_NODES:
            kind, cls_or_leaf, child_jsons = UNSUPPORTED_RECORD
        else:
            kind, cls_or_leaf, child_jsons = dispatch_parser(curr)
            ipc_fanout = (
                len(cls_or_leaf.values) if isinstance(cls_or_leaf, ListLiteral) else 0
            )
            if (
                len(raw_nodes) + ipc_leaf_nodes + len(child_jsons) + ipc_fanout
                > MAX_AST_NODES
            ):
                # Degrade only the oversized subtree to Unsupported() rather than
                # aborting the entire root AST so shallow conjunctive (AND) filters
                # already discovered in BFS order can still be pushed down.
                kind, cls_or_leaf, child_jsons = UNSUPPORTED_RECORD
                ipc_fanout = 0
            ipc_leaf_nodes += ipc_fanout

        child_indices: list[int] = []
        for child_json in child_jsons:
            child_idx = len(raw_nodes)
            raw_nodes.append(child_json)
            queue.append(child_idx)
            child_indices.append(child_idx)

        records.append((kind, cls_or_leaf, tuple(child_indices)))

    # Pass 2: Bottom-up IR tree construction.
    # Because Pass 1 guarantees `idx < c_idx` for all `c_idx` in `child_indices`,
    # iterating from `len(records) - 1` down to `0` ensures that `ir_nodes[c_idx]`
    # is already populated with a constructed frozen `Expr` dataclass before its
    # parent at `idx` is instantiated. `ir_nodes[0]` is the root `Expr`.
    ir_nodes: list[Expr] = [Unsupported()] * len(records)
    for idx in range(len(records) - 1, -1, -1):
        kind, cls_or_leaf, child_indices = records[idx]
        if kind == "Leaf" and isinstance(cls_or_leaf, Expr):
            ir_nodes[idx] = cls_or_leaf
        elif kind == "Unary" and len(child_indices) == 1:
            ir_nodes[idx] = cls_or_leaf(expr=ir_nodes[child_indices[0]])
        elif kind == "Binary" and len(child_indices) == 2:
            ir_nodes[idx] = cls_or_leaf(
                left=ir_nodes[child_indices[0]],
                right=ir_nodes[child_indices[1]],
            )
        elif kind == "Variadic":
            ir_nodes[idx] = cls_or_leaf(
                tuple(ir_nodes[c_idx] for c_idx in child_indices)
            )
        else:
            ir_nodes[idx] = Unsupported(
                operands=tuple(ir_nodes[c_idx] for c_idx in child_indices)
            )

    return ir_nodes[0]


__all__ = [
    "PARSERS",
    "base",
    "boolean",
    "comparison",
    "dispatch_parser",
    "json_to_ir",
    "list_",
    "numeric",
    "parse_json_path",
    "register_parser",
    "string",
    "temporal",
]
