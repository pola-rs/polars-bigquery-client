from __future__ import annotations

import collections
import functools

from polars_bigquery.core.compiler.ir.base import (
    Expr,
    NullLiteral,
    Unsupported,
)
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    Coalesce,
    DropNulls,
    IsNotNull,
    Not,
    Or,
    Ternary,
    When,
)
from polars_bigquery.core.compiler.rewriter.base import replace_children, rewrite_tree

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


def rewrite_boolean_when_node(node: Expr) -> Expr:
    """Rewrite a `Ternary` node into boolean conditionals when its output is used as a boolean.

    In a positive boolean context:
    - If both `truthy` and `falsy` are falsy (`False` or `Null`), rewrites to `False`.
    - If `falsy` is falsy, rewrites `Ternary(p, t, f)` to `p & t`.
    - If `truthy` is falsy, rewrites `Ternary(p, t, f)` to `~COALESCE(p, False) & f`.
    - If `node` contains an `Unsupported` descendant, rewrites `Ternary(p, t, f)` to
      `(p & t) | (~COALESCE(p, False) & f)` to enable partial pushdown.
    """
    if not isinstance(node, Ternary) or not _is_valid_boolean_ternary(node):
        return node
    truthy_is_falsy = node.truthy in _FALSY_LITERALS
    falsy_is_falsy = node.falsy in _FALSY_LITERALS
    if truthy_is_falsy and falsy_is_falsy:
        return BoolLiteral(value=False)
    if falsy_is_falsy:
        return And(left=node.predicate, right=node.truthy)
    negated_pred: Expr = Not(
        expr=Coalesce(operands=(node.predicate, BoolLiteral(value=False)))
    )
    if truthy_is_falsy:
        return And(left=negated_pred, right=node.falsy)
    if node.has_descendant(Unsupported):
        if node.predicate.has_descendant(Unsupported):
            negated_pred = Unsupported()
        return Or(
            left=And(left=node.predicate, right=node.truthy),
            right=And(left=negated_pred, right=node.falsy),
        )
    return node


def rewrite_boolean_when(root: Expr) -> Expr:
    """Rewrite boolean-valued `Ternary` IR nodes across an IR tree into boolean conditionals.

    Propagates boolean context top-down by node index from the filter root
    through `And`, `Or`, and `Ternary` predicates and branches, then rewrites
    boolean-context `Ternary` nodes bottom-up.
    """
    nodes: list[Expr] = [root]
    children_map: list[tuple[int, ...]] = [()]
    boolean_nodes: set[int] = {0}
    queue: collections.deque[int] = collections.deque([0])

    while queue:
        curr_idx = queue.popleft()
        node = nodes[curr_idx]
        node_children = node.children()
        if not node_children:
            continue

        child_indices: list[int] = []
        for child in node_children:
            c_idx = len(nodes)
            nodes.append(child)
            children_map.append(())
            queue.append(c_idx)
            child_indices.append(c_idx)
        children_map[curr_idx] = tuple(child_indices)

        in_boolean_context = curr_idx in boolean_nodes
        if isinstance(node, (And, Or)) and in_boolean_context:
            boolean_nodes.update(child_indices)
        elif isinstance(node, Ternary):
            boolean_nodes.add(child_indices[0])
            if in_boolean_context and _is_valid_boolean_ternary(node):
                boolean_nodes.update(child_indices[1:])

    for idx in range(len(nodes) - 1, -1, -1):
        child_indices = children_map[idx]
        if child_indices:
            child_nodes = tuple(nodes[c_idx] for c_idx in child_indices)
            nodes[idx] = replace_children(nodes[idx], child_nodes)
        if idx in boolean_nodes:
            nodes[idx] = rewrite_boolean_when_node(nodes[idx])

    return nodes[0]


def rewrite_boolean_literals_node(node: Expr) -> Expr:
    """Collapse `And` and `Or` nodes when an operand is a `BoolLiteral`."""
    if isinstance(node, And):
        if node.left == BoolLiteral(False) or node.right == BoolLiteral(False):
            return BoolLiteral(value=False)
        if node.left == BoolLiteral(True):
            return node.right
        if node.right == BoolLiteral(True):
            return node.left

    if isinstance(node, Or):
        if node.left == BoolLiteral(True) or node.right == BoolLiteral(True):
            return BoolLiteral(value=True)
        if node.left == BoolLiteral(False):
            return node.right
        if node.right == BoolLiteral(False):
            return node.left

    return node


def rewrite_boolean_literals(root: Expr) -> Expr:
    """Collapse `And` and `Or` nodes with boolean literal operands across an IR tree."""
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
    """Rewrite negated `Or` nodes into `And` of negations via De Morgan's law.

    Handles both `Not(Or(left, right))` and null-safe negated predicates
    `Not(Coalesce((Or(left, right), BoolLiteral(False))))`. We intentionally
    only apply De Morgan's law in the direction that turns `Or` into `And`
    (and not `Not(And(...))` into `Or(...)`), because `And` allows either
    branch to contain an `Unsupported` descendant while still preserving
    predicate pushdown for the supported branch.
    """
    if isinstance(node, Not):
        if isinstance(node.expr, Or):
            return And(
                left=Not(expr=node.expr.left),
                right=Not(expr=node.expr.right),
            )
        if (
            isinstance(node.expr, Coalesce)
            and len(node.expr.operands) == 2
            and isinstance(node.expr.operands[0], Or)
            and node.expr.operands[1] == BoolLiteral(value=False)
        ):
            or_node = node.expr.operands[0]
            return And(
                left=Not(
                    expr=Coalesce(operands=(or_node.left, BoolLiteral(value=False)))
                ),
                right=Not(
                    expr=Coalesce(operands=(or_node.right, BoolLiteral(value=False)))
                ),
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
