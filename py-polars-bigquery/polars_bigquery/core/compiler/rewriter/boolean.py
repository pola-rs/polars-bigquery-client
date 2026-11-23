from __future__ import annotations

from polars_bigquery.core.compiler._traversal import flatten_expr_bfs
from polars_bigquery.core.compiler.ir.base import (
    Expr,
    NullLiteral,
    Unsupported,
)
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    Coalesce,
    Not,
    Or,
    Ternary,
    When,
)
from polars_bigquery.core.compiler.ir.list_ import ListLiteral
from polars_bigquery.core.compiler.rewriter.base import replace_children, rewrite_tree

_FALSY_LITERALS = (BoolLiteral(False), NullLiteral())
MAX_BOOLEAN_WHEN_REWRITE_NODES = 20_000


def _node_self_weight(node: Expr) -> int:
    """Return the intrinsic AST weight of a single IR node, weighting `ListLiteral` by fanout."""
    if isinstance(node, ListLiteral):
        return 1 + len(node.values)
    return 1


def _is_de_morgan_not(node: Not) -> bool:
    """Return True if `rewrite_de_morgan_node` can split `node` into an `And`."""
    if isinstance(node.expr, Or):
        return True
    return (
        isinstance(node.expr, Coalesce)
        and len(node.expr.operands) == 2
        and isinstance(node.expr.operands[0], Or)
        and node.expr.operands[1] == BoolLiteral(value=False)
    )


def _is_trivial_truthy_or_unsup(node: Expr) -> bool:
    """Return True if evaluating `node` in a filter imposes no SQL row restriction.

    Iterative single-node check matching `rewrite_boolean_when`'s bottom-up
    `unconstrained` tracker:
    - `Unsupported` / `BoolLiteral(True)` impose no restriction.
    - `And(l, r)` is unconstrained iff both `l` and `r` are unconstrained.
    - `Or(l, r)` is unconstrained iff either `l` or `r` is unconstrained (`emit_or_sql`
      drops an entire `Or` when either child is `None`/`Unsupported` or `TRUE`).
    - Non-monotone operators containing an `Unsupported` descendant emit `(None, False)`
      in SQL (except `Not` shapes split by `rewrite_de_morgan` into `And`).
    """
    nodes, children_map = flatten_expr_bfs(node)
    n = len(nodes)
    flags = [False] * n
    has_unsup = [False] * n
    for idx in range(n - 1, -1, -1):
        curr = nodes[idx]
        c_indices = children_map[idx]
        node_unsup = isinstance(curr, Unsupported) or any(
            has_unsup[c_idx] for c_idx in c_indices
        )
        has_unsup[idx] = node_unsup
        if isinstance(curr, Unsupported) or curr == BoolLiteral(True):
            flags[idx] = True
        elif isinstance(curr, And):
            left_idx, right_idx = c_indices
            flags[idx] = flags[left_idx] and flags[right_idx]
        elif isinstance(curr, Or):
            left_idx, right_idx = c_indices
            flags[idx] = flags[left_idx] or flags[right_idx]
        elif isinstance(curr, Not):
            flags[idx] = node_unsup and not _is_de_morgan_not(curr)
        else:
            flags[idx] = node_unsup
    return flags[0]


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


def _make_negated_pred(predicate: Expr) -> Expr:
    """Build `NOT COALESCE(predicate, FALSE)` preserving 3-valued NULL fallthrough."""
    return Not(expr=Coalesce(operands=(predicate, BoolLiteral(value=False))))


def _rewrite_boolean_ternary_step(
    node: Ternary,
    *,
    has_unsup: bool,
    pred_unsup: bool,
    truthy_unconstrained: bool,
    falsy_unconstrained: bool,
) -> tuple[Expr, bool]:
    """Shared single-node lowering for a validated boolean-context `Ternary`.

    Returns `(rewritten_expr, needs_budget_duplication)` where the rewritten node
    is one of:
    - `BoolLiteral(False)` when both branches are falsy (`False` / `Null`).
    - `And(p, t)` when `falsy` is falsy.
    - `And(NOT COALESCE(p, False), f)` when `truthy` is falsy.
    - `BoolLiteral(True)` when both branches are `BoolLiteral(True)` and the
      predicate has no `Unsupported` descendant.
    - `Unsupported()` when both branches are trivial (`True` / unconstrained) and
      the node contains an `Unsupported` descendant (avoiding emitting tautological
      `p OR NOT COALESCE(p, FALSE)` filters that prune zero rows).
    - `Or(And(p, t), And(NOT COALESCE(p, False) | Unsupported, f))` when the node
      contains an `Unsupported` descendant.
    - `node` unchanged otherwise.
    """
    truthy_is_falsy = node.truthy in _FALSY_LITERALS
    falsy_is_falsy = node.falsy in _FALSY_LITERALS
    if truthy_is_falsy and falsy_is_falsy:
        return BoolLiteral(value=False), False
    if falsy_is_falsy:
        return And(left=node.predicate, right=node.truthy), False
    if truthy_is_falsy:
        return And(left=_make_negated_pred(node.predicate), right=node.falsy), False
    if (
        not pred_unsup
        and node.truthy == BoolLiteral(True)
        and node.falsy == BoolLiteral(True)
    ):
        return BoolLiteral(value=True), False
    if has_unsup:
        if truthy_unconstrained and falsy_unconstrained:
            return Unsupported(), False
        negated_pred: Expr = (
            Unsupported() if pred_unsup else _make_negated_pred(node.predicate)
        )
        return (
            Or(
                left=And(left=node.predicate, right=node.truthy),
                right=And(left=negated_pred, right=node.falsy),
            ),
            True,
        )
    return node, False


def rewrite_boolean_when_node(node: Expr) -> Expr:
    """Rewrite a `Ternary` node into boolean conditionals when its output is used as a boolean.

    In a positive boolean context:
    - If both `truthy` and `falsy` are falsy (`False` or `Null`), rewrites to `False`.
    - If `falsy` is falsy, rewrites `Ternary(p, t, f)` to `p & t`.
    - If `truthy` is falsy, rewrites `Ternary(p, t, f)` to `~COALESCE(p, False) & f`.
    - If both `truthy` and `falsy` are `True` and `predicate` contains no
      `Unsupported` descendant, rewrites to `True`.
    - If both `truthy` and `falsy` are trivial (`True` / unconstrained) and `node`
      contains an `Unsupported` descendant, degrades to `Unsupported` (avoiding
      tautological `p OR ~COALESCE(p, False)` pushdown).
    - If `node` contains an `Unsupported` descendant, rewrites `Ternary(p, t, f)` to
      `(p & t) | (~COALESCE(p, False) & f)` to enable partial pushdown.
    """
    if not isinstance(node, Ternary) or not _is_valid_boolean_ternary(node):
        return node
    has_unsup = node.has_descendant(Unsupported)
    pred_unsup = has_unsup and node.predicate.has_descendant(Unsupported)
    rewritten, _ = _rewrite_boolean_ternary_step(
        node,
        has_unsup=has_unsup,
        pred_unsup=pred_unsup,
        truthy_unconstrained=has_unsup and _is_trivial_truthy_or_unsup(node.truthy),
        falsy_unconstrained=has_unsup and _is_trivial_truthy_or_unsup(node.falsy),
    )
    return rewritten


def rewrite_boolean_when(root: Expr) -> Expr:
    """Rewrite boolean-valued `Ternary` IR nodes across an IR tree in linear $O(N)$ time.

    Uses three iterative passes over `flatten_expr_bfs(root)` (zero Python recursion):
    1. Bottom-up pass over original nodes computing `orig_ok` / `orig_has_leaf` for
       every `Ternary` chain in $O(1)$ per node (avoiding quadratic re-walks).
    2. Top-down pass propagating boolean-context flags from the filter root through
       `And`, `Or`, and `Ternary` predicates/branches.
    3. Bottom-up reconstruction rewriting validated `Ternary` nodes with shared
       cumulative AST expansion budget `MAX_BOOLEAN_WHEN_REWRITE_NODES` (accounting
       for `ListLiteral` fanout weight `1 + len(values)`).
    """
    nodes, children_map = flatten_expr_bfs(root)
    n = len(nodes)

    # Pass 1: Precompute `_is_valid_boolean_ternary` flags bottom-up in O(N).
    orig_ok = [True] * n
    orig_leaf = [False] * n
    for idx in range(n - 1, -1, -1):
        node = nodes[idx]
        if isinstance(node, Ternary):
            p_idx, t_idx, f_idx = children_map[idx]
            orig_ok[idx] = (
                not isinstance(nodes[p_idx], NullLiteral)
                and orig_ok[t_idx]
                and orig_ok[f_idx]
            )
            orig_leaf[idx] = orig_leaf[t_idx] or orig_leaf[f_idx]
        else:
            orig_leaf[idx] = not isinstance(node, NullLiteral)

    # Pass 2: Propagate boolean context top-down in BFS index order (`parent < child`).
    boolean_nodes = [False] * n
    boolean_nodes[0] = True
    for idx in range(n):
        c_indices = children_map[idx]
        if not c_indices:
            continue
        node = nodes[idx]
        if isinstance(node, (And, Or)) and boolean_nodes[idx]:
            for c_idx in c_indices:
                boolean_nodes[c_idx] = True
        elif isinstance(node, Ternary):
            boolean_nodes[c_indices[0]] = True
            if boolean_nodes[idx] and orig_ok[idx] and orig_leaf[idx]:
                boolean_nodes[c_indices[1]] = True
                boolean_nodes[c_indices[2]] = True

    # Pass 3: Bottom-up reconstruction with O(1) validity & global expansion budget.
    total_input_weight = sum(_node_self_weight(node) for node in nodes)
    remaining_budget = max(0, MAX_BOOLEAN_WHEN_REWRITE_NODES - total_input_weight)

    weights = [_node_self_weight(node) for node in nodes]
    has_unsup = [isinstance(node, Unsupported) for node in nodes]
    unconstrained = [
        isinstance(node, Unsupported) or node == BoolLiteral(True) for node in nodes
    ]
    post_ok = [True] * n
    post_leaf = [not isinstance(node, NullLiteral) for node in nodes]

    for idx in range(n - 1, -1, -1):
        c_indices = children_map[idx]
        if c_indices:
            child_nodes = tuple(nodes[c_idx] for c_idx in c_indices)
            reconstructed = replace_children(nodes[idx], child_nodes)
            nodes[idx] = reconstructed
            weights[idx] = _node_self_weight(reconstructed) + sum(
                weights[c_idx] for c_idx in c_indices
            )
            has_unsup[idx] = isinstance(reconstructed, Unsupported) or any(
                has_unsup[c_idx] for c_idx in c_indices
            )
            if isinstance(reconstructed, And):
                unconstrained[idx] = (
                    unconstrained[c_indices[0]] and unconstrained[c_indices[1]]
                )
            elif isinstance(reconstructed, Or):
                unconstrained[idx] = (
                    unconstrained[c_indices[0]] or unconstrained[c_indices[1]]
                )
            elif isinstance(reconstructed, Not):
                unconstrained[idx] = has_unsup[idx] and not _is_de_morgan_not(
                    reconstructed
                )
            else:
                unconstrained[idx] = has_unsup[idx] or (
                    reconstructed == BoolLiteral(True)
                )
            if isinstance(reconstructed, Ternary):
                p_idx, t_idx, f_idx = c_indices
                post_ok[idx] = (
                    not isinstance(nodes[p_idx], NullLiteral)
                    and post_ok[t_idx]
                    and post_ok[f_idx]
                )
                post_leaf[idx] = post_leaf[t_idx] or post_leaf[f_idx]
            else:
                post_ok[idx] = True
                post_leaf[idx] = not isinstance(reconstructed, NullLiteral)

        if boolean_nodes[idx]:
            curr = nodes[idx]
            if isinstance(curr, Ternary) and post_ok[idx] and post_leaf[idx]:
                p_idx, t_idx, f_idx = c_indices
                pred_unsup = has_unsup[p_idx]
                pred_w = weights[p_idx]
                truthy_w = weights[t_idx]
                falsy_w = weights[f_idx]
                orig_w = weights[idx]

                rewritten, duplicates_pred = _rewrite_boolean_ternary_step(
                    curr,
                    has_unsup=has_unsup[idx],
                    pred_unsup=pred_unsup,
                    truthy_unconstrained=unconstrained[t_idx],
                    falsy_unconstrained=unconstrained[f_idx],
                )
                if rewritten is curr:
                    continue
                if isinstance(rewritten, BoolLiteral):
                    nodes[idx] = rewritten
                    weights[idx] = 1
                    has_unsup[idx] = False
                    unconstrained[idx] = rewritten.value
                    post_ok[idx] = True
                    post_leaf[idx] = True
                    continue
                if isinstance(rewritten, Unsupported):
                    nodes[idx] = rewritten
                    weights[idx] = 1
                    has_unsup[idx] = True
                    unconstrained[idx] = True
                    post_ok[idx] = True
                    post_leaf[idx] = True
                    continue
                if not duplicates_pred:
                    # `falsy_is_falsy` -> `And(p, t)` or `truthy_is_falsy` -> `And(Not(Coalesce(p, False)), f)`
                    if curr.falsy in _FALSY_LITERALS:
                        nodes[idx] = rewritten
                        weights[idx] = 1 + pred_w + truthy_w
                        has_unsup[idx] = pred_unsup or has_unsup[t_idx]
                        unconstrained[idx] = (
                            unconstrained[p_idx] and unconstrained[t_idx]
                        )
                    else:
                        # Replacing `Ternary(p, falsy_lit, f)` (`2 + p + f` nodes) with
                        # `And(Not(Coalesce(p, False)), f)` (`4 + p + f` nodes) adds 2 nodes.
                        extra_cost = 2
                        if (
                            4 + pred_w + falsy_w > MAX_BOOLEAN_WHEN_REWRITE_NODES
                            or extra_cost > remaining_budget
                        ):
                            nodes[idx] = Unsupported()
                            weights[idx] = 1
                            has_unsup[idx] = True
                            unconstrained[idx] = True
                        else:
                            remaining_budget -= extra_cost
                            nodes[idx] = rewritten
                            weights[idx] = 4 + pred_w + falsy_w
                            has_unsup[idx] = pred_unsup or has_unsup[f_idx]
                            neg_unconstrained = pred_unsup and not isinstance(
                                curr.predicate, Or
                            )
                            unconstrained[idx] = (
                                neg_unconstrained and unconstrained[f_idx]
                            )
                    post_ok[idx] = True
                    post_leaf[idx] = True
                    continue

                neg_w = 1 if pred_unsup else (pred_w + 3)
                new_w = 3 + pred_w + truthy_w + neg_w + falsy_w
                extra_w = max(0, new_w - orig_w)
                if new_w > MAX_BOOLEAN_WHEN_REWRITE_NODES or extra_w > remaining_budget:
                    nodes[idx] = Unsupported()
                    weights[idx] = 1
                    has_unsup[idx] = True
                    unconstrained[idx] = True
                else:
                    remaining_budget -= extra_w
                    nodes[idx] = rewritten
                    weights[idx] = new_w
                    has_unsup[idx] = True
                    left_unconstrained = unconstrained[p_idx] and unconstrained[t_idx]
                    right_unconstrained = pred_unsup and unconstrained[f_idx]
                    unconstrained[idx] = left_unconstrained or right_unconstrained
                post_ok[idx] = True
                post_leaf[idx] = True

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
        # `rewrite_boolean_when_node` wraps negated Ternary predicates in
        # `Coalesce(p, False)` so NULL predicates evaluate to TRUE when negated.
        # Distribute `Coalesce(..., False)` across `Or` operands so De Morgan's
        # law can still split negated `Or` predicates into `And` while
        # preserving 3-valued logic when either operand is NULL:
        # NOT COALESCE(a OR b, FALSE) <=> NOT COALESCE(a, FALSE) AND NOT COALESCE(b, FALSE)
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
