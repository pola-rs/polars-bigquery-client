from __future__ import annotations

import functools
from collections.abc import Iterator

from polars_bigquery.core.compiler.ir.base import (
    Expr,
    Literal,
    NullLiteral,
    Unsupported,
)
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    DropNulls,
    IsNotNull,
    Not,
    Or,
    Ternary,
    When,
)
from polars_bigquery.core.compiler.ir.list_ import ListLiteral
from polars_bigquery.core.compiler.ir.string import Lowercase, Uppercase
from polars_bigquery.core.compiler.rewriter.base import rewrite_tree


def _is_non_boolean_branch_leaf(expr: Expr) -> bool:
    """Return True if `expr` is a known non-boolean leaf or transform."""
    return (
        isinstance(expr, Literal) and not isinstance(expr, (BoolLiteral, NullLiteral))
    ) or isinstance(expr, (Lowercase, Uppercase))


def _is_valid_boolean_ternary(node: Ternary) -> bool:
    """Return True if `node` and any nested `Ternary` branches form a valid boolean conditional."""
    stack: list[Expr] = [node]
    has_non_null_leaf = False
    while stack:
        curr = stack.pop()
        if isinstance(curr, Ternary):
            if isinstance(curr.predicate, (NullLiteral, ListLiteral)):
                return False
            stack.append(curr.falsy)
            stack.append(curr.truthy)
        else:
            if _is_non_boolean_branch_leaf(curr):
                return False
            if not isinstance(curr, NullLiteral):
                has_non_null_leaf = True
    return has_non_null_leaf


def _iter_ternary_leaves(expr: Expr) -> Iterator[Expr]:
    """Yield the non-`Ternary` branch leaves of `expr`."""
    stack: list[Expr] = [expr]
    while stack:
        curr = stack.pop()
        if isinstance(curr, Ternary):
            stack.append(curr.falsy)
            stack.append(curr.truthy)
        else:
            yield curr


def _is_falsy_branch(expr: Expr) -> bool:
    """Return True if `expr` (or every branch of a nested `Ternary`) is `False` or `NULL`."""
    return all(
        leaf in (NullLiteral(), BoolLiteral(False))
        for leaf in _iter_ternary_leaves(expr)
    )


def _is_truthy_branch(expr: Expr) -> bool:
    """Return True if `expr` (or every branch of a nested `Ternary`) is `True`."""
    return all(leaf == BoolLiteral(True) for leaf in _iter_ternary_leaves(expr))


def _rewrite_ternary_to_boolean_expr(node: Ternary) -> Expr:
    """Convert a validated boolean `Ternary(p, t, f)` into `And` / `Or` / `Not` IR nodes."""
    p, t, f = node.predicate, node.truthy, node.falsy
    if _is_falsy_branch(t) or _is_truthy_branch(f):
        p, t, f = Not(expr=p), f, t
    if _is_falsy_branch(t):
        return BoolLiteral(value=False)
    if _is_truthy_branch(f):
        return BoolLiteral(value=True)
    if _is_falsy_branch(f):
        return p if _is_truthy_branch(t) else And(left=p, right=t)
    if _is_truthy_branch(t):
        return Or(left=p, right=f)
    return Or(
        left=And(left=p, right=t),
        right=And(left=Not(expr=p), right=f),
    )


def rewrite_boolean_when_node(node: Expr, boolean_nodes: set[int]) -> Expr:
    """Rewrite a `Ternary` node into boolean conditionals when its output is used as a boolean.

    Propagates boolean context top-down from the filter root through `And`, `Or`,
    and `Ternary` predicates and branches, then rewrites boolean-valued
    `Ternary(p, t, f)` nodes into `(p & t) | (~p & f)` (with constant-folding
    simplifications when `t` or `f` is `True`, `False`, or `NULL`).
    """
    in_boolean_context = id(node) in boolean_nodes

    if isinstance(node, (And, Or)) and in_boolean_context:
        boolean_nodes.update((id(node.left), id(node.right)))
        return node

    if isinstance(node, Ternary):
        boolean_nodes.add(id(node.predicate))
        if not in_boolean_context or not _is_valid_boolean_ternary(node):
            return node
        rewritten = _rewrite_ternary_to_boolean_expr(node)
        boolean_nodes.update((id(node.truthy), id(node.falsy), id(rewritten)))
        return rewritten

    return node


def rewrite_boolean_when(root: Expr) -> Expr:
    """Rewrite boolean-valued `Ternary` IR nodes across an IR tree into boolean conditionals."""
    boolean_nodes: set[int] = {id(root)}
    return rewrite_tree(
        root,
        functools.partial(rewrite_boolean_when_node, boolean_nodes=boolean_nodes),
    )


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


def _strip_drop_nulls_node(node: Expr) -> Expr:
    """Strip a `DropNulls` wrapper by returning its inner expression."""
    return node.expr if isinstance(node, DropNulls) else node


def rewrite_drop_nulls_node(node: Expr, *, not_null_exprs: list[Expr]) -> Expr:
    """Unwrap a single `DropNulls` node and record its inner expression for `IS NOT NULL`."""
    if not isinstance(node, DropNulls):
        return node
    if node.expr.has_descendant(Unsupported):
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
