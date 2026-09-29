from __future__ import annotations

import collections
import dataclasses
import functools
from collections.abc import Callable

from polars_bigquery.core.compiler.ir.base import (
    BinaryExpr,
    Expr,
    TernaryExpr,
    UnaryExpr,
    Unsupported,
)

NodeRewriterFunc = Callable[[Expr], Expr]
RewriterFunc = Callable[[Expr], Expr]


@functools.singledispatch
def replace_children(node: Expr, children: tuple[Expr, ...]) -> Expr:
    """Return a copy of `node` with its children replaced by `children`."""
    if not children:
        return node
    msg = f"Cannot replace children on leaf or unhandled IR node: {node!r}"
    raise TypeError(msg)


@replace_children.register
def replace_unary_children(node: UnaryExpr, children: tuple[Expr, ...]) -> Expr:
    """Return a copy of a `UnaryExpr` node with its single child replaced."""
    if len(children) != 1:
        msg = f"UnaryExpr requires exactly 1 child, got {len(children)}"
        raise ValueError(msg)
    if children[0] is node.expr:
        return node
    return dataclasses.replace(node, expr=children[0])


@replace_children.register
def replace_binary_children(node: BinaryExpr, children: tuple[Expr, ...]) -> Expr:
    """Return a copy of a `BinaryExpr` node with its two children replaced."""
    if len(children) != 2:
        msg = f"BinaryExpr requires exactly 2 children, got {len(children)}"
        raise ValueError(msg)
    if children[0] is node.left and children[1] is node.right:
        return node
    return dataclasses.replace(node, left=children[0], right=children[1])


@replace_children.register
def replace_ternary_children(node: TernaryExpr, children: tuple[Expr, ...]) -> Expr:
    """Return a copy of a `TernaryExpr` node with its three children replaced."""
    if len(children) != 3:
        msg = f"TernaryExpr requires exactly 3 children, got {len(children)}"
        raise ValueError(msg)
    if (
        children[0] is node.predicate
        and children[1] is node.truthy
        and children[2] is node.falsy
    ):
        return node
    return dataclasses.replace(
        node, predicate=children[0], truthy=children[1], falsy=children[2]
    )


@replace_children.register
def replace_unsupported_children(node: Unsupported, children: tuple[Expr, ...]) -> Expr:
    """Return a copy of an `Unsupported` node with its operands replaced."""
    if len(children) == len(node.operands) and all(
        new_c is old_c for new_c, old_c in zip(children, node.operands, strict=False)
    ):
        return node
    return dataclasses.replace(node, operands=children)


def rewrite_tree(root: Expr, node_rewriter: NodeRewriterFunc) -> Expr:
    """Rewrite an `Expr` IR tree iteratively using `node_rewriter`.

    Performs a two-pass iterative traversal to avoid Python's recursion limit
    on deeply nested expression trees:
    1. Pass 1 (Top-down BFS traversal): Applies `node_rewriter` to each node until
       it reaches a fixed point before queueing its children, allowing top-down
       rewrites (such as pushing `Not` down through nested `Or` nodes) to cascade.
    2. Pass 2 (Bottom-up reconstruction): Reconstructs parent nodes with their
       rewritten children in reverse BFS order.
    """
    nodes: list[Expr] = [root]
    children_map: list[tuple[int, ...]] = [()]
    queue: collections.deque[int] = collections.deque([0])

    while queue:
        curr_idx = queue.popleft()
        node = nodes[curr_idx]
        while True:
            rewritten = node_rewriter(node)
            if rewritten is node or rewritten == node:
                break
            node = rewritten
        nodes[curr_idx] = node

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

    for idx in range(len(nodes) - 1, -1, -1):
        child_indices = children_map[idx]
        if child_indices:
            child_nodes = tuple(nodes[c_idx] for c_idx in child_indices)
            nodes[idx] = replace_children(nodes[idx], child_nodes)

    return nodes[0]
