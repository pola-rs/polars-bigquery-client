from __future__ import annotations

import collections
import functools

from polars_bigquery.core.compiler.ir.base import Expr, Unsupported
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    DropNulls,
    IsNotNull,
    Not,
    Or,
    Ternary,
    When,
)
from polars_bigquery.core.compiler.rewriter.base import rewrite_tree


def rewrite_when_node(node: Expr) -> Expr:
    """Collapse nested `Ternary` nodes in `falsy` into a `When` node.

    Polars serializes `.when(p1).then(t1).when(p2).then(t2).otherwise(f)`
    expressions as nested `Ternary` nodes where the second clause is the `falsy`
    child of the outer `Ternary`. Standalone `Ternary` nodes remain `Ternary`
    (so SQL generation can emit `IF(...)`), whereas chained `Ternary` nodes in
    the `falsy` branch collapse into a variadic `When` node (`CASE WHEN ... END`).
    """
    if not isinstance(node, Ternary) or not isinstance(node.falsy, Ternary):
        return node
    operands: list[Expr] = [node.predicate, node.truthy]
    curr: Expr = node.falsy
    while isinstance(curr, Ternary):
        operands.append(curr.predicate)
        operands.append(curr.truthy)
        curr = curr.falsy
    operands.append(curr)
    return When(operands=tuple(operands))


def rewrite_when(root: Expr) -> Expr:
    """Collapse chained `Ternary` IR nodes across an IR tree into multi-clause `When` nodes."""
    return rewrite_tree(root, rewrite_when_node)


def rewrite_de_morgan_node(node: Expr) -> Expr:
    """Rewrite `Not(Or(left, right))` into `And(Not(left), Not(right))`.

    We intentionally only apply De Morgan's law in the direction that turns `Or`
    into `And` (and not `Not(And(...))` into `Or(...)`), because `And` allows
    either branch to contain an `Unsupported` descendant while still preserving
    predicate pushdown for the supported branch.
    """
    if isinstance(node, Not) and isinstance(node.expr, Or):
        return And(
            left=Not(expr=node.expr.left),
            right=Not(expr=node.expr.right),
        )
    return node


def rewrite_de_morgan(root: Expr) -> Expr:
    """Apply De Morgan's law across an IR tree to turn negated `Or` nodes into `And` nodes."""
    return rewrite_tree(root, rewrite_de_morgan_node)


def _has_unsupported_descendant(root: Expr) -> bool:
    """Return True if `root` or any of its descendants is `Unsupported`."""
    queue: collections.deque[Expr] = collections.deque([root])
    while queue:
        node = queue.popleft()
        if isinstance(node, Unsupported):
            return True
        queue.extend(node.children())
    return False


def _strip_drop_nulls_node(node: Expr) -> Expr:
    """Strip a `DropNulls` wrapper by returning its inner expression."""
    return node.expr if isinstance(node, DropNulls) else node


def rewrite_drop_nulls_node(node: Expr, *, not_null_exprs: list[Expr]) -> Expr:
    """Unwrap a single `DropNulls` node and record its inner expression for `IS NOT NULL`."""
    if not isinstance(node, DropNulls):
        return node
    if _has_unsupported_descendant(node.expr):
        return Unsupported()
    not_null_exprs.append(node.expr)
    return node.expr


def rewrite_drop_nulls(root: Expr) -> Expr:
    """Rewrite `DropNulls(expr)` into `expr` and conjoin `IsNotNull(expr)` at the root.

    If any descendant of a `DropNulls` node is `Unsupported`, the entire
    `DropNulls` node is rewritten to `Unsupported` without adding a top-level
    `IsNotNull` check.
    """
    not_null_exprs: list[Expr] = []
    node_rewriter = functools.partial(
        rewrite_drop_nulls_node,
        not_null_exprs=not_null_exprs,
    )
    result = rewrite_tree(root, node_rewriter)
    for expr in not_null_exprs:
        cleaned_expr = rewrite_tree(expr, _strip_drop_nulls_node)
        result = And(left=result, right=IsNotNull(expr=cleaned_expr))
    return result
