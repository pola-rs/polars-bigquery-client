"""Typing and lowering handlers for boolean, null-handling, and conditional IR nodes."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from polars_bigquery.core.compiler.ir.base import Expr, Literal, Unsupported
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    Coalesce,
    DropNulls,
    FillNull,
    FillNullWithStrategy,
    IsNotNull,
    IsNull,
    Not,
    Or,
    Ternary,
    When,
)
from polars_bigquery.core.compiler.ir.list_ import ListLiteral
from polars_bigquery.core.compiler.ir.numeric import (
    DecimalLiteral,
    FloatLiteral,
    IntLiteral,
)
from polars_bigquery.core.compiler.ir.string import BytesLiteral, StringLiteral
from polars_bigquery.core.compiler.ir.temporal import (
    DateLiteral,
    TimeLiteral,
    TimestampLiteral,
    TimestampUnit,
)
from polars_bigquery.core.compiler.typing.base import (
    TimeUnitStr,
    _degrade,
    _is_bool_logical_operand,
    _is_exact_typed_operand,
    _is_null_branch,
    _is_valid_predicate,
    _type_node,
    _TypedNode,
    _TypingContext,
    _unify_many,
    _unify_types,
)
from polars_bigquery.core.types import OPAQUE_BQ_TYPES, BQType

logger = logging.getLogger(__name__)


def _fill_null_strategy_literal(
    bq_type: BQType,
    time_unit: TimeUnitStr | None,
    strategy: str,
    *,
    decimal_supports_one: bool = True,
) -> Literal | None:
    """Resolve the typed GoogleSQL literal for `FillNullWithStrategy`, or None."""
    if strategy not in ("Zero", "One"):
        return None
    is_one = strategy == "One"
    if bq_type is BQType.BOOL:
        return BoolLiteral(value=is_one)
    if bq_type is BQType.INT64:
        return IntLiteral(value=1 if is_one else 0)
    if bq_type is BQType.FLOAT64:
        return FloatLiteral(value=1.0 if is_one else 0.0)
    if bq_type is BQType.NUMERIC:
        if is_one and not decimal_supports_one:
            return None
        return DecimalLiteral(value="1" if is_one else "0")
    if bq_type is BQType.DATE:
        return DateLiteral(days=1 if is_one else 0)
    if bq_type is BQType.STRING:
        return None if is_one else StringLiteral(value="")
    if bq_type is BQType.BYTES:
        return None if is_one else BytesLiteral(value=b"")
    if bq_type is BQType.TIME:
        return None if is_one else TimeLiteral(micros=0)
    if bq_type in (BQType.DATETIME, BQType.TIMESTAMP):
        tz = None if bq_type is BQType.DATETIME else "UTC"
        if not is_one:
            return TimestampLiteral(ticks=0, unit=TimestampUnit.MICROSECONDS, tz=tz)
        if time_unit == "ms":
            return TimestampLiteral(ticks=1, unit=TimestampUnit.MILLISECONDS, tz=tz)
        if time_unit == "us":
            return TimestampLiteral(ticks=1, unit=TimestampUnit.MICROSECONDS, tz=tz)
        return None
    # Fail closed on any opaque or newly-added BQType member.
    return None


def _type_conditional(
    node: Ternary | When,
    predicates: Sequence[_TypedNode],
    branches: Sequence[_TypedNode],
    all_children: Sequence[_TypedNode],
    ctx: _TypingContext,
) -> _TypedNode:
    """Shared typing logic for `Ternary` and defensive `When` IR nodes."""
    if not all(_is_valid_predicate(pred) for pred in predicates):
        logger.debug(
            "Conditional %r has non-BOOL or NULL predicate; degrading to Unsupported",
            type(node).__name__,
        )
        return _degrade(all_children, ctx)

    if any(
        isinstance(branch.node, ListLiteral) or branch.bq_type in OPAQUE_BQ_TYPES
        for branch in branches
    ):
        logger.debug(
            "Conditional %r has list or opaque value branch; degrading to Unsupported",
            type(node).__name__,
        )
        return _degrade(all_children, ctx)

    reconstructed: Expr = (
        Ternary(
            predicate=all_children[0].node,
            truthy=all_children[1].node,
            falsy=all_children[2].node,
        )
        if isinstance(node, Ternary)
        else When(operands=tuple(child.node for child in all_children))
    )
    weight = 1 + sum(child.weight for child in all_children)
    has_unsupported = any(child.has_unsupported for child in all_children)
    total_expansion_cost = sum(child.expansion_cost for child in all_children)

    if all(_is_null_branch(branch) for branch in branches):
        return _TypedNode(
            node=reconstructed,
            bq_type=None,
            time_unit=None,
            is_known_non_nan=True,
            is_known_non_null=False,
            weight=weight,
            has_unsupported=has_unsupported,
            is_all_null=True,
            expansion_cost=total_expansion_cost,
        )

    if any(isinstance(branch.node, Unsupported) for branch in branches):
        known_branches = [
            branch
            for branch in branches
            if not _is_null_branch(branch) and not isinstance(branch.node, Unsupported)
        ]
        # If every single branch is Unsupported (no known BOOL branch and no NULL
        # branch), the conditional would only produce a tautology (`p OR NOT
        # COALESCE(p, FALSE)`) or mis-type non-boolean expressions (`when(p).then(x*2).otherwise(x*3)`).
        has_null_branch = any(_is_null_branch(branch) for branch in branches)
        if not known_branches and not has_null_branch:
            logger.debug(
                "Conditional %r has only Unsupported value branches; degrading to Unsupported",
                type(node).__name__,
            )
            return _degrade(all_children, ctx)
        if all(branch.bq_type is BQType.BOOL for branch in known_branches):
            return _TypedNode(
                node=reconstructed,
                bq_type=BQType.BOOL,
                time_unit=None,
                is_known_non_nan=True,
                is_known_non_null=False,
                weight=weight,
                has_unsupported=True,
                expansion_cost=total_expansion_cost,
            )
        logger.debug(
            "Conditional %r with Unsupported value branch has non-BOOL sibling branch; degrading to Unsupported",
            type(node).__name__,
        )
        return _degrade(all_children, ctx)

    non_null_branches = [branch for branch in branches if not _is_null_branch(branch)]
    non_null_types: list[BQType] = []
    for branch in non_null_branches:
        if branch.bq_type is None:
            logger.debug(
                "Conditional %r has untyped non-NULL branch; degrading to Unsupported",
                type(node).__name__,
            )
            return _degrade(all_children, ctx)
        non_null_types.append(branch.bq_type)

    unified = _unify_many(non_null_types)
    if unified is None:
        logger.debug(
            "Conditional %r value branches have incompatible types; degrading to Unsupported",
            type(node).__name__,
        )
        return _degrade(all_children, ctx)

    time_units = {branch.time_unit for branch in non_null_branches}
    time_unit = next(iter(time_units)) if len(time_units) == 1 else None
    is_known_non_nan = all(branch.is_known_non_nan for branch in non_null_branches)
    # Under IF / CASE WHEN, a NULL predicate falls through to the next WHEN / ELSE
    # branch rather than yielding NULL, so non-nullness depends only on the value branches.
    is_known_non_null = not has_unsupported and all(
        branch.is_known_non_null for branch in branches
    )
    decimal_supports_one = all(
        branch.decimal_supports_one for branch in non_null_branches
    )
    return _TypedNode(
        node=reconstructed,
        bq_type=unified,
        time_unit=time_unit,
        is_known_non_nan=is_known_non_nan,
        is_known_non_null=is_known_non_null,
        weight=weight,
        has_unsupported=has_unsupported,
        decimal_supports_one=decimal_supports_one,
        expansion_cost=total_expansion_cost,
    )


@_type_node.register(And)
@_type_node.register(Or)
def _type_binary_logical(
    node: And | Or,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    left, right = children
    if not (_is_bool_logical_operand(left) and _is_bool_logical_operand(right)):
        logger.debug(
            "Logical operator %s received non-BOOL operand types (%r, %r); degrading to Unsupported",
            type(node).__name__,
            left.bq_type,
            right.bq_type,
        )
        return _degrade(children, ctx)

    # All-NULL conditional operands (`when(p).then(None).otherwise(None)`) cannot
    # emit valid SQL; lowering them to `Unsupported()` lets `And` keep partial
    # pushdown for the valid sibling while degrading `Or` cleanly.
    left_node = left.node
    left_unsup = left.has_unsupported
    left_cost = left.expansion_cost
    if left.is_all_null:
        ctx.refund_budget(left.expansion_cost)
        left_node = Unsupported()
        left_unsup = True
        left_cost = 0

    right_node = right.node
    right_unsup = right.has_unsupported
    right_cost = right.expansion_cost
    if right.is_all_null:
        ctx.refund_budget(right.expansion_cost)
        right_node = Unsupported()
        right_unsup = True
        right_cost = 0

    has_unsupported = left_unsup or right_unsup
    return _TypedNode(
        node=type(node)(left=left_node, right=right_node),
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=(
            not has_unsupported and left.is_known_non_null and right.is_known_non_null
        ),
        weight=1
        + (1 if left.is_all_null else left.weight)
        + (1 if right.is_all_null else right.weight),
        has_unsupported=has_unsupported,
        expansion_cost=left_cost + right_cost,
    )


@_type_node.register
def _type_not(
    _node: Not,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    (operand,) = children
    if not _is_bool_logical_operand(operand):
        logger.debug(
            "Logical NOT received non-BOOL operand type %r; degrading to Unsupported",
            operand.bq_type,
        )
        return _degrade(children, ctx)
    if operand.is_all_null:
        return _degrade(children, ctx)
    return _TypedNode(
        node=Not(expr=operand.node),
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=not operand.has_unsupported and operand.is_known_non_null,
        weight=1 + operand.weight,
        has_unsupported=operand.has_unsupported,
        expansion_cost=operand.expansion_cost,
    )


@_type_node.register(IsNull)
@_type_node.register(IsNotNull)
def _type_null_check(
    node: IsNull | IsNotNull,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    (operand,) = children
    if not _is_exact_typed_operand(operand):
        logger.debug(
            "%s received invalid operand %r; degrading to Unsupported",
            type(node).__name__,
            operand.node,
        )
        return _degrade(children, ctx)
    return _TypedNode(
        node=type(node)(expr=operand.node),
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=True,
        weight=1 + operand.weight,
        has_unsupported=False,
        expansion_cost=operand.expansion_cost,
    )


@_type_node.register
def _type_drop_nulls(
    _node: DropNulls,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    """Degrade `DropNulls` to `Unsupported()` to prevent silent row loss.

    In Polars, `expr.drop_nulls()` changes Series length and broadcasts length-1
    results back across the entire DataFrame (including null rows). Hoisting
    `expr IS NOT NULL` to BigQuery would drop matching null rows when a column
    has exactly 1 non-null value and silently suppress Polars `ShapeError` when
    it has `>= 2` non-null values alongside nulls.
    """
    logger.debug(
        "DropNulls is not pushed down because series-length broadcasting cannot "
        "guarantee a row-level superset; falling back to client-side evaluation"
    )
    return _degrade(children, ctx)


@_type_node.register
def _type_fill_null_with_strategy(
    node: FillNullWithStrategy,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    (operand,) = children
    if (
        not _is_exact_typed_operand(operand)
        or operand.bq_type is None
        or operand.bq_type in OPAQUE_BQ_TYPES
    ):
        logger.debug(
            "FillNullWithStrategy(%r) received invalid/opaque operand type %r; degrading to Unsupported",
            node.strategy,
            operand.bq_type,
        )
        return _degrade(children, ctx)

    fill_literal = _fill_null_strategy_literal(
        operand.bq_type,
        operand.time_unit,
        node.strategy,
        decimal_supports_one=operand.decimal_supports_one,
    )
    if fill_literal is None:
        logger.debug(
            "FillNullWithStrategy(%r) is not representable for type %r (time_unit=%r); degrading to Unsupported",
            node.strategy,
            operand.bq_type,
            operand.time_unit,
        )
        return _degrade(children, ctx)

    if not ctx.consume_budget(1):
        logger.debug(
            "AST node budget exhausted lowering FillNullWithStrategy(%r); degrading to Unsupported",
            node.strategy,
        )
        return _degrade(children, ctx)

    return _TypedNode(
        node=FillNull(left=operand.node, right=fill_literal),
        bq_type=operand.bq_type,
        time_unit=operand.time_unit,
        is_known_non_nan=operand.is_known_non_nan,
        is_known_non_null=True,
        weight=operand.weight + 2,
        has_unsupported=False,
        decimal_supports_one=operand.decimal_supports_one,
        expansion_cost=operand.expansion_cost + 1,
    )


@_type_node.register
def _type_fill_null(
    _node: FillNull,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    left, right = children
    if (
        not _is_exact_typed_operand(left)
        or not _is_exact_typed_operand(right)
        or left.bq_type is None
        or right.bq_type is None
    ):
        return _degrade(children, ctx)
    unified = _unify_types(left.bq_type, right.bq_type)
    if unified is None:
        logger.debug(
            "FillNull operands have incompatible types (%r, %r); degrading to Unsupported",
            left.bq_type,
            right.bq_type,
        )
        return _degrade(children, ctx)
    time_unit = left.time_unit if left.time_unit == right.time_unit else None
    return _TypedNode(
        node=FillNull(left=left.node, right=right.node),
        bq_type=unified,
        time_unit=time_unit,
        is_known_non_nan=left.is_known_non_nan and right.is_known_non_nan,
        is_known_non_null=left.is_known_non_null or right.is_known_non_null,
        weight=1 + left.weight + right.weight,
        has_unsupported=False,
        decimal_supports_one=left.decimal_supports_one and right.decimal_supports_one,
        expansion_cost=left.expansion_cost + right.expansion_cost,
    )


@_type_node.register
def _type_coalesce(
    _node: Coalesce,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    if not children or any(not _is_exact_typed_operand(child) for child in children):
        return _degrade(children, ctx)
    child_types: list[BQType] = []
    for child in children:
        if child.bq_type is None:
            return _degrade(children, ctx)
        child_types.append(child.bq_type)
    unified = _unify_many(child_types)
    if unified is None:
        logger.debug(
            "Coalesce operands have incompatible or opaque types; degrading to Unsupported"
        )
        return _degrade(children, ctx)
    time_units = {child.time_unit for child in children}
    time_unit = next(iter(time_units)) if len(time_units) == 1 else None
    return _TypedNode(
        node=Coalesce(operands=tuple(child.node for child in children)),
        bq_type=unified,
        time_unit=time_unit,
        is_known_non_nan=all(child.is_known_non_nan for child in children),
        is_known_non_null=any(child.is_known_non_null for child in children),
        weight=1 + sum(child.weight for child in children),
        has_unsupported=False,
        decimal_supports_one=all(child.decimal_supports_one for child in children),
        expansion_cost=sum(child.expansion_cost for child in children),
    )


@_type_node.register
def _type_ternary(
    node: Ternary,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    pred, truthy, falsy = children
    return _type_conditional(node, (pred,), (truthy, falsy), children, ctx)


@_type_node.register
def _type_when(
    node: When,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    if len(children) < 3 or len(children) % 2 == 0:
        logger.debug(
            "When node has invalid arity %d; degrading to Unsupported",
            len(children),
        )
        return _degrade(children, ctx)
    predicates = children[:-1:2]
    branches = (*children[1::2], children[-1])
    return _type_conditional(node, predicates, branches, children, ctx)
