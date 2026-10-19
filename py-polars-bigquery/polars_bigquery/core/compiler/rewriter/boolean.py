from __future__ import annotations

import functools

from polars_bigquery.core.compiler.ir.base import (
    Expr,
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
from polars_bigquery.core.compiler.rewriter.base import rewrite_tree

_FALSY_LITERALS = (BoolLiteral(False), NullLiteral())


def _is_valid_boolean_ternary(node: Ternary) -> bool:
    """Return True if `node` and any nested `Ternary` branches form a valid boolean conditional."""
    stack: list[Expr] = [node]
    has_non_null_leaf = False
    while stack:
        curr = stack.pop()
        if isinstance(curr, Ternary):
            if isinstance(curr.predicate, NullLiteral):
                return False
            stack.append(curr.falsy)
            stack.append(curr.truthy)
        elif not isinstance(curr, NullLiteral):
            has_non_null_leaf = True
    return has_non_null_leaf


def rewrite_boolean_when_node(node: Expr, boolean_nodes: set[int]) -> Expr:
    """Rewrite a `Ternary` node into boolean conditionals when its output is used as a boolean.

    Propagates boolean context top-down from the filter root through `And`, `Or`,
    and `Ternary` predicates and branches, then rewrites boolean-valued
    `Ternary(p, t, f)` nodes into `(p & t) | (~p & f)`.
    """
    in_boolean_context = id(node) in boolean_nodes

    if isinstance(node, (And, Or)) and in_boolean_context:
        boolean_nodes.update((id(node.left), id(node.right)))
        return node

    if isinstance(node, Ternary):
        boolean_nodes.add(id(node.predicate))
        if not in_boolean_context or not _is_valid_boolean_ternary(node):
            return node
        rewritten = Or(
            left=And(left=node.predicate, right=node.truthy),
            right=And(left=Not(expr=node.predicate), right=node.falsy),
        )
        boolean_nodes.add(id(rewritten))
        return rewritten

    return node


def rewrite_boolean_when(root: Expr) -> Expr:
    """Rewrite boolean-valued `Ternary` IR nodes across an IR tree into boolean conditionals."""
    boolean_nodes: set[int] = {id(root)}
    return rewrite_tree(
        root,
        functools.partial(rewrite_boolean_when_node, boolean_nodes=boolean_nodes),
    )


def rewrite_boolean_literals_node(node: Expr) -> Expr:
    """Collapse `And` and `Or` nodes when an operand is a `BoolLiteral` or `NullLiteral`."""
    if isinstance(node, And):
        if node.left in _FALSY_LITERALS or node.right in _FALSY_LITERALS:
            return BoolLiteral(value=False)
        if node.left == BoolLiteral(True):
            return node.right
        if node.right == BoolLiteral(True):
            return node.left

    if isinstance(node, Or):
        if node.left == BoolLiteral(True) or node.right == BoolLiteral(True):
            return BoolLiteral(value=True)
        if node.left in _FALSY_LITERALS:
            return node.right
        if node.right in _FALSY_LITERALS:
            return node.left

    return node


def rewrite_boolean_literals(root: Expr) -> Expr:
    """Collapse `And` and `Or` nodes with boolean or null literal operands across an IR tree."""
    return rewrite_tree(root, rewrite_boolean_literals_node)


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
