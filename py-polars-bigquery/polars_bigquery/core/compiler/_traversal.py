"""Shared iterative BFS traversal helper for compiler IR trees."""

from __future__ import annotations

import collections

from polars_bigquery.core.compiler.ir.base import Expr


def flatten_expr_bfs(root: Expr) -> tuple[list[Expr], list[tuple[int, ...]]]:
    """Flatten an `Expr` tree in top-down BFS order without Python recursion.

    Returns `(nodes, children_map)` such that every child edge `(parent, child)`
    satisfies `parent < child`. Evaluating `nodes` from index `len(nodes) - 1`
    down to `0` therefore guarantees a bottom-up post-order evaluation.
    """
    nodes: list[Expr] = [root]
    children_map: list[tuple[int, ...]] = [()]
    queue: collections.deque[int] = collections.deque([0])

    while queue:
        curr_idx = queue.popleft()
        node_children = nodes[curr_idx].children()
        if node_children:
            child_indices: list[int] = []
            for child in node_children:
                c_idx = len(nodes)
                nodes.append(child)
                children_map.append(())
                queue.append(c_idx)
                child_indices.append(c_idx)
            children_map[curr_idx] = tuple(child_indices)

    return nodes, children_map


__all__ = ["flatten_expr_bfs"]
