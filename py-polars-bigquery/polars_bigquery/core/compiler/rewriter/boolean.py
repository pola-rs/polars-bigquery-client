from __future__ import annotations

import dataclasses

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


def _is_node_unconstrained(
    node: Expr,
    child_unconstrained: tuple[bool, ...],
    *,
    has_unsup: bool,
) -> bool:
    """Return True if `node` imposes no SQL row restriction when used as a filter.

    Mirrors SQL emission (`sql/boolean.py`) and the subsequent `rewrite_de_morgan`
    pass:
    - `And(l, r)` is unconstrained iff both `l` and `r` are unconstrained.
    - `Or(l, r)` is unconstrained iff either `l` or `r` is unconstrained
      (`emit_or_sql` drops the entire `Or` if either side is `None`/`TRUE`).
    - `Not(x)` with an `Unsupported` descendant emits `(None, False)` **except**
      for De Morgan-splittable shapes (`Not(Or(...))` and
      `Not(Coalesce((Or(...), False)))`), which `rewrite_de_morgan` turns into
      an `And` that can still push down supported branches.
    - All other operators containing an `Unsupported` descendant emit
      `(None, False)` in SQL; leaves are unconstrained iff `Unsupported` or
      `BoolLiteral(True)`.
    """
    if isinstance(node, And):
        return child_unconstrained[0] and child_unconstrained[1]
    if isinstance(node, Or):
        return child_unconstrained[0] or child_unconstrained[1]
    if isinstance(node, Not):
        return has_unsup and not _is_de_morgan_not(node)
    return has_unsup or node == BoolLiteral(True)


def _is_trivial_truthy_or_unsup(node: Expr) -> bool:
    """Return True if evaluating `node` in a filter imposes no SQL row restriction.

    Iterative single-node check matching `rewrite_boolean_when`'s bottom-up
    `unconstrained` tracker via `_is_node_unconstrained`.
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
        child_flags = tuple(flags[c_idx] for c_idx in c_indices)
        flags[idx] = _is_node_unconstrained(curr, child_flags, has_unsup=node_unsup)
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


def _compute_valid_ternary_flags(
    nodes: list[Expr], children_map: list[tuple[int, ...]]
) -> list[bool]:
    """Precompute `_is_valid_boolean_ternary` bottom-up in O(N) over a BFS node array.

    Returns a list aligned with `nodes` where `result[idx]` is `True` iff the
    subtree rooted at `nodes[idx]` satisfies `_is_valid_boolean_ternary` (no
    `NullLiteral` predicate anywhere in its `Ternary` branch chain, and at least
    one non-`NullLiteral` leaf across its branches).
    """
    n = len(nodes)
    ok = [True] * n
    has_leaf = [False] * n
    for idx in range(n - 1, -1, -1):
        node = nodes[idx]
        if isinstance(node, Ternary):
            p_idx, t_idx, f_idx = children_map[idx]
            ok[idx] = (
                not isinstance(nodes[p_idx], NullLiteral) and ok[t_idx] and ok[f_idx]
            )
            has_leaf[idx] = has_leaf[t_idx] or has_leaf[f_idx]
        else:
            has_leaf[idx] = not isinstance(node, NullLiteral)
    return [ok[idx] and has_leaf[idx] for idx in range(n)]


def _propagate_boolean_context(
    nodes: list[Expr],
    children_map: list[tuple[int, ...]],
    valid_ternary: list[bool],
) -> list[bool]:
    """Propagate positive boolean-context flags top-down in BFS order (`parent < child`).

    The root expression (index `0`) is always in a filter/boolean context.
    - Under `And`/`Or` in a boolean context, both children inherit boolean context.
    - Under `Ternary`, the predicate is always boolean; `truthy`/`falsy` branches
      inherit boolean context only when the `Ternary` itself is in a boolean context
      and forms a valid boolean conditional (`valid_ternary[idx]`).
    - Other operators (e.g. `Eq`, `Coalesce`, `IsNull`) switch children to scalar
      context (`False`), so nested `when/then/otherwise` outputs are preserved.
    """
    boolean_nodes = [False] * len(nodes)
    boolean_nodes[0] = True
    for idx, node in enumerate(nodes):
        c_indices = children_map[idx]
        if not c_indices:
            continue
        if isinstance(node, (And, Or)) and boolean_nodes[idx]:
            for c_idx in c_indices:
                boolean_nodes[c_idx] = True
        elif isinstance(node, Ternary):
            boolean_nodes[c_indices[0]] = True
            if boolean_nodes[idx] and valid_ternary[idx]:
                boolean_nodes[c_indices[1]] = True
                boolean_nodes[c_indices[2]] = True
    return boolean_nodes


@dataclasses.dataclass(frozen=True, slots=True)
class _RewriteSlot:
    """Synthesized bottom-up attributes for a single BFS slot in `rewrite_boolean_when`.

    Storing each node together with its synthesized metadata in one immutable record
    guarantees all attributes are updated atomically whenever a node is rebuilt or
    rewritten, avoiding stale side-table bugs while keeping every query O(1).

    Attributes:
        expr: Current `Expr` at this BFS slot (updated as children are rebuilt/rewritten).
        weight: Total AST weight of `expr`, where `ListLiteral` contributes
            `1 + len(values)` and all other nodes contribute `1`.
        has_unsup: `True` if `expr` or any descendant is an `Unsupported` node.
        unconstrained: `True` if `expr` imposes no BigQuery SQL row restriction
            in a filter context (matching `_is_node_unconstrained`).
        is_ok: `True` if no `Ternary` in `expr`'s output branch chain has a
            `NullLiteral` predicate (always `True` for non-`Ternary` roots).
        has_non_null_leaf: `True` if `expr`'s output branch chain contains at least
            one non-`NullLiteral` leaf.
    """

    expr: Expr
    weight: int
    has_unsup: bool
    unconstrained: bool
    is_ok: bool = True
    has_non_null_leaf: bool = True


_UNSUPPORTED_SLOT = _RewriteSlot(
    expr=Unsupported(),
    weight=1,
    has_unsup=True,
    unconstrained=True,
)


@dataclasses.dataclass(slots=True)
class _BooleanWhenRewriteState:
    """Pass-3 bottom-up state and expansion budget for `rewrite_boolean_when`.

    Maintains one `_RewriteSlot` per BFS node index alongside a single shared
    `remaining_budget` (derived from `MAX_BOOLEAN_WHEN_REWRITE_NODES` minus the
    total initial input weight). Every slot transitions through up to two stages
    during the reverse-BFS (`len(slots) - 1` down to `0`) sweep:

    1. `reconstruct(idx, child_indices)`: Rebuilds interior node `idx` from its
       already-finalized child slots and recomputes `weight`, `has_unsup`,
       `unconstrained`, `is_ok`, and `has_non_null_leaf` in O(arity).
    2. `try_rewrite_ternary(idx, child_indices)`: If slot `idx` is in boolean
       context and is a valid `Ternary`, lowers it via `_rewrite_boolean_ternary_step`,
       deducting any AST weight growth from `remaining_budget` (or degrading the
       slot to `_UNSUPPORTED_SLOT` if either `MAX_BOOLEAN_WHEN_REWRITE_NODES` or
       `remaining_budget` would be exceeded).

    See `docs/designs/polars_bigquery_boolean_when_rewriter.md` for full design
    rationale, invariants, and trade-offs.
    """

    slots: list[_RewriteSlot]
    remaining_budget: int

    @classmethod
    def from_nodes(cls, nodes: list[Expr]) -> _BooleanWhenRewriteState:
        """Initialize leaf-level slots and global rewrite budget from BFS `nodes`."""
        slots = [
            _RewriteSlot(
                expr=node,
                weight=_node_self_weight(node),
                has_unsup=isinstance(node, Unsupported),
                unconstrained=isinstance(node, Unsupported)
                or node == BoolLiteral(True),
                is_ok=True,
                has_non_null_leaf=not isinstance(node, NullLiteral),
            )
            for node in nodes
        ]
        total_input_weight = sum(slot.weight for slot in slots)
        remaining_budget = max(0, MAX_BOOLEAN_WHEN_REWRITE_NODES - total_input_weight)
        return cls(slots=slots, remaining_budget=remaining_budget)

    def _try_consume_budget(self, new_weight: int, extra_weight: int) -> bool:
        """Reserve `extra_weight` nodes against both subtree and global growth caps.

        Shrinking rewrites intentionally do not refund `remaining_budget`, keeping
        budget consumption monotonic and deterministic across the sweep.
        """
        if (
            new_weight > MAX_BOOLEAN_WHEN_REWRITE_NODES
            or extra_weight > self.remaining_budget
        ):
            return False
        self.remaining_budget -= extra_weight
        return True

    def reconstruct(self, idx: int, child_indices: tuple[int, ...]) -> None:
        """Rebuild interior slot `idx` from already-finalized child slots in O(arity)."""
        children = tuple(self.slots[c_idx] for c_idx in child_indices)
        reconstructed = replace_children(
            self.slots[idx].expr, tuple(child.expr for child in children)
        )
        weight = _node_self_weight(reconstructed) + sum(
            child.weight for child in children
        )
        node_unsup = isinstance(reconstructed, Unsupported) or any(
            child.has_unsup for child in children
        )
        unconstrained = _is_node_unconstrained(
            reconstructed,
            tuple(child.unconstrained for child in children),
            has_unsup=node_unsup,
        )
        if isinstance(reconstructed, Ternary):
            pred, truthy, falsy = children
            is_ok = (
                not isinstance(pred.expr, NullLiteral) and truthy.is_ok and falsy.is_ok
            )
            has_leaf = truthy.has_non_null_leaf or falsy.has_non_null_leaf
        else:
            is_ok = True
            has_leaf = not isinstance(reconstructed, NullLiteral)

        self.slots[idx] = _RewriteSlot(
            expr=reconstructed,
            weight=weight,
            has_unsup=node_unsup,
            unconstrained=unconstrained,
            is_ok=is_ok,
            has_non_null_leaf=has_leaf,
        )

    def _apply_falsy_truthy_rewrite(
        self,
        idx: int,
        rewritten: Expr,
        pred: _RewriteSlot,
        falsy: _RewriteSlot,
    ) -> None:
        """Record `Ternary(p, falsy_lit, f)` -> `And(Not(Coalesce(p, False)), f)`.

        Replaces a `(2 + p + f)`-weight subtree with a `(4 + p + f)`-weight subtree
        (`extra_weight = 2`). Note that a De Morgan-splittable `Or` predicate is
        treated as constrained because `rewrite_de_morgan` splits
        `Not(Coalesce(Or(l, r), False))` into an `And` of negations.
        """
        new_w = 4 + pred.weight + falsy.weight
        if not self._try_consume_budget(new_w, extra_weight=2):
            self.slots[idx] = _UNSUPPORTED_SLOT
            return
        neg_unconstrained = pred.has_unsup and not isinstance(pred.expr, Or)
        self.slots[idx] = _RewriteSlot(
            expr=rewritten,
            weight=new_w,
            has_unsup=pred.has_unsup or falsy.has_unsup,
            unconstrained=neg_unconstrained and falsy.unconstrained,
        )

    def _apply_duplicated_pred_rewrite(
        self,
        idx: int,
        rewritten: Expr,
        pred: _RewriteSlot,
        truthy: _RewriteSlot,
        falsy: _RewriteSlot,
    ) -> None:
        """Record `Ternary(p, t, f)` -> `Or(And(p, t), And(neg_p, f))` within budget.

        When `pred.has_unsup` is `True`, `neg_p` is `Unsupported()` (weight 1);
        otherwise `neg_p` duplicates `p` inside `Not(Coalesce((p, False)))`
        (weight `pred.weight + 3`), charging the full `ListLiteral` fanout.
        """
        neg_w = 1 if pred.has_unsup else (pred.weight + 3)
        new_w = 3 + pred.weight + truthy.weight + neg_w + falsy.weight
        extra_w = max(0, new_w - self.slots[idx].weight)
        if not self._try_consume_budget(new_w, extra_w):
            self.slots[idx] = _UNSUPPORTED_SLOT
            return
        left_unconstrained = pred.unconstrained and truthy.unconstrained
        right_unconstrained = pred.has_unsup and falsy.unconstrained
        self.slots[idx] = _RewriteSlot(
            expr=rewritten,
            weight=new_w,
            has_unsup=True,
            unconstrained=left_unconstrained or right_unconstrained,
        )

    def try_rewrite_ternary(self, idx: int, child_indices: tuple[int, ...]) -> None:
        """Lower a validated boolean-context `Ternary` at `idx`, enforcing the budget."""
        slot = self.slots[idx]
        curr = slot.expr
        if not (isinstance(curr, Ternary) and slot.is_ok and slot.has_non_null_leaf):
            return

        p_idx, t_idx, f_idx = child_indices
        pred, truthy, falsy = (
            self.slots[p_idx],
            self.slots[t_idx],
            self.slots[f_idx],
        )
        rewritten, duplicates_pred = _rewrite_boolean_ternary_step(
            curr,
            has_unsup=slot.has_unsup,
            pred_unsup=pred.has_unsup,
            truthy_unconstrained=truthy.unconstrained,
            falsy_unconstrained=falsy.unconstrained,
        )
        if rewritten is curr:
            return
        if isinstance(rewritten, BoolLiteral):
            self.slots[idx] = _RewriteSlot(
                expr=rewritten,
                weight=1,
                has_unsup=False,
                unconstrained=rewritten.value,
            )
            return
        if isinstance(rewritten, Unsupported):
            self.slots[idx] = _UNSUPPORTED_SLOT
            return
        if duplicates_pred:
            self._apply_duplicated_pred_rewrite(idx, rewritten, pred, truthy, falsy)
            return
        if curr.falsy in _FALSY_LITERALS:
            # `Ternary(p, t, False|Null)` -> `And(p, t)` shrinks weight by 1 node.
            self.slots[idx] = _RewriteSlot(
                expr=rewritten,
                weight=1 + pred.weight + truthy.weight,
                has_unsup=pred.has_unsup or truthy.has_unsup,
                unconstrained=pred.unconstrained and truthy.unconstrained,
            )
            return
        self._apply_falsy_truthy_rewrite(idx, rewritten, pred, falsy)


def rewrite_boolean_when(root: Expr) -> Expr:
    """Rewrite boolean-valued `Ternary` IR nodes across an IR tree in linear $O(N)$ time.

    Uses three iterative passes over `flatten_expr_bfs(root)` (zero Python recursion):
    1. Bottom-up pass over original nodes computing `_is_valid_boolean_ternary` for
       every `Ternary` chain in $O(1)$ per node (avoiding quadratic re-walks).
    2. Top-down pass propagating boolean-context flags from the filter root through
       `And`, `Or`, and `Ternary` predicates/branches.
    3. Bottom-up reconstruction rewriting validated `Ternary` nodes via
       `_BooleanWhenRewriteState` with cumulative AST expansion budget
       `MAX_BOOLEAN_WHEN_REWRITE_NODES` (weighting `ListLiteral` as `1 + len(values)`).
    """
    nodes, children_map = flatten_expr_bfs(root)
    valid_ternary = _compute_valid_ternary_flags(nodes, children_map)
    boolean_nodes = _propagate_boolean_context(nodes, children_map, valid_ternary)

    state = _BooleanWhenRewriteState.from_nodes(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        c_indices = children_map[idx]
        if c_indices:
            state.reconstruct(idx, c_indices)
        if boolean_nodes[idx]:
            state.try_rewrite_ternary(idx, c_indices)

    return state.slots[0].expr


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
