"""Typing and lowering for numeric predicates and `FLOAT64` `NaN` total-order comparisons."""

from __future__ import annotations

import logging
import math

from polars_bigquery.core.compiler.ir.base import Expr, NullLiteral
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    IsNotNull,
    Not,
    Or,
    Ternary,
)
from polars_bigquery.core.compiler.ir.comparison import (
    Eq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    NotEq,
)
from polars_bigquery.core.compiler.ir.numeric import (
    FloatLiteral,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
)
from polars_bigquery.core.compiler.typing.base import (
    MAX_DUPLICATED_OPERAND_NODES,
    _degrade,
    _is_exact_typed_operand,
    _type_node,
    _TypedNode,
    _TypingContext,
)
from polars_bigquery.core.types import BQType

logger = logging.getLogger(__name__)

ComparisonExpr = Eq | NotEq | Gt | GtEq | Lt | LtEq

_FLIPPED_COMPARISON_CLS: dict[type[ComparisonExpr], type[ComparisonExpr]] = {
    Eq: Eq,
    NotEq: NotEq,
    Gt: Lt,
    GtEq: LtEq,
    Lt: Gt,
    LtEq: GtEq,
}


def _build_null_safe_bool_const(
    operand: _TypedNode,
    *,
    const_value: bool,
    extra_weight: int,
    ctx: _TypingContext,
) -> tuple[Expr, int, int] | None:
    """Build `Ternary(IsNotNull(operand), BoolLiteral(const_value), NullLiteral())`.

    Returns `(expr, new_weight, extra_weight_consumed)` or `None` if budget is exhausted.
    """
    if not ctx.consume_budget(extra_weight):
        return None
    expr = Ternary(
        predicate=IsNotNull(expr=operand.node),
        truthy=BoolLiteral(value=const_value),
        falsy=NullLiteral(),
    )
    return expr, operand.weight + 4, extra_weight


def _lower_nan_literal_comparison(
    op_cls: type[ComparisonExpr],
    other: _TypedNode,
    ctx: _TypingContext,
) -> tuple[Expr, int, int] | None:
    """Lower `other op FloatLiteral(NaN)` (Case A) to exact 3-valued GoogleSQL IR."""
    if op_cls in (Eq, GtEq):
        if other.is_known_non_nan:
            return _build_null_safe_bool_const(
                other, const_value=False, extra_weight=2, ctx=ctx
            )
        return IsNan(expr=other.node), other.weight + 1, 0
    if op_cls in (NotEq, Lt):
        if other.is_known_non_nan:
            return _build_null_safe_bool_const(
                other, const_value=True, extra_weight=2, ctx=ctx
            )
        return Not(expr=IsNan(expr=other.node)), other.weight + 2, 0
    if op_cls is LtEq:
        return _build_null_safe_bool_const(
            other, const_value=True, extra_weight=2, ctx=ctx
        )
    if op_cls is Gt:
        return _build_null_safe_bool_const(
            other, const_value=False, extra_weight=2, ctx=ctx
        )
    return None


def _lower_one_sided_nan_comparison(
    cmp_expr: ComparisonExpr,
    orig_weight: int,
    *,
    nan_side: _TypedNode,
    non_nan_side: _TypedNode,
    nan_on_left: bool,
    ctx: _TypingContext,
) -> tuple[Expr, int, int] | None:
    """Lower Case B / Case C when exactly one side can be `NaN` and the operator needs NaN handling."""
    if non_nan_side.is_known_non_null:
        # Case B: duplicates only `nan_side`.
        extra_weight = 2 + nan_side.weight
        if nan_side.weight > MAX_DUPLICATED_OPERAND_NODES or not ctx.consume_budget(
            extra_weight
        ):
            return None
        return (
            Or(left=cmp_expr, right=IsNan(expr=nan_side.node)),
            orig_weight + extra_weight,
            extra_weight,
        )

    # Case C: duplicates both `nan_side` and `non_nan_side`.
    extra_weight = 4 + nan_side.weight + non_nan_side.weight
    if (
        nan_side.weight > MAX_DUPLICATED_OPERAND_NODES
        or non_nan_side.weight > MAX_DUPLICATED_OPERAND_NODES
        or not ctx.consume_budget(extra_weight)
    ):
        return None
    nan_guard = (
        And(left=IsNan(expr=nan_side.node), right=IsNotNull(expr=non_nan_side.node))
        if nan_on_left
        else And(
            left=IsNotNull(expr=non_nan_side.node), right=IsNan(expr=nan_side.node)
        )
    )
    return Or(left=cmp_expr, right=nan_guard), orig_weight + extra_weight, extra_weight


def _lower_two_sided_nan_comparison(
    cmp_expr: ComparisonExpr,
    orig_weight: int,
    left: _TypedNode,
    right: _TypedNode,
    ctx: _TypingContext,
) -> tuple[Expr, int, int] | None:
    """Lower Case D when both `left` and `right` can be `NaN`."""
    if (
        left.weight > MAX_DUPLICATED_OPERAND_NODES
        or right.weight > MAX_DUPLICATED_OPERAND_NODES
    ):
        return None

    if isinstance(cmp_expr, Eq):
        extra_weight = 4 + left.weight + right.weight
        lowered: Expr = Or(
            left=cmp_expr,
            right=And(left=IsNan(expr=left.node), right=IsNan(expr=right.node)),
        )
    elif isinstance(cmp_expr, NotEq):
        extra_weight = 5 + left.weight + right.weight
        lowered = And(
            left=cmp_expr,
            right=Not(
                expr=And(left=IsNan(expr=left.node), right=IsNan(expr=right.node))
            ),
        )
    elif isinstance(cmp_expr, Gt):
        extra_weight = 5 + left.weight + right.weight
        lowered = Or(
            left=cmp_expr,
            right=And(
                left=IsNan(expr=left.node),
                right=Not(expr=IsNan(expr=right.node)),
            ),
        )
    elif isinstance(cmp_expr, GtEq):
        extra_weight = 4 + left.weight + right.weight
        lowered = Or(
            left=cmp_expr,
            right=And(left=IsNan(expr=left.node), right=IsNotNull(expr=right.node)),
        )
    elif isinstance(cmp_expr, Lt):
        extra_weight = 5 + left.weight + right.weight
        lowered = Or(
            left=cmp_expr,
            right=And(
                left=Not(expr=IsNan(expr=left.node)),
                right=IsNan(expr=right.node),
            ),
        )
    elif isinstance(cmp_expr, LtEq):
        extra_weight = 4 + left.weight + right.weight
        lowered = Or(
            left=cmp_expr,
            right=And(left=IsNotNull(expr=left.node), right=IsNan(expr=right.node)),
        )
    else:
        return None

    if not ctx.consume_budget(extra_weight):
        return None
    return lowered, orig_weight + extra_weight, extra_weight


def _lower_float_comparison(
    node: ComparisonExpr,
    left: _TypedNode,
    right: _TypedNode,
    ctx: _TypingContext,
) -> tuple[Expr, int, int] | None:
    """Lower a FLOAT64 comparison to preserve Polars' NaN total order under 3VL."""
    orig_weight = 1 + left.weight + right.weight
    cmp_expr: ComparisonExpr = type(node)(left=left.node, right=right.node)

    # Case 0: neither operand can be NaN.
    if left.is_known_non_nan and right.is_known_non_nan:
        return cmp_expr, orig_weight, 0

    # Case A: FloatLiteral(NaN) on either side.
    if isinstance(right.node, FloatLiteral) and math.isnan(right.node.value):
        return _lower_nan_literal_comparison(type(node), left, ctx)
    if isinstance(left.node, FloatLiteral) and math.isnan(left.node.value):
        flipped_cls = _FLIPPED_COMPARISON_CLS.get(type(node))
        if flipped_cls is None:
            return None
        return _lower_nan_literal_comparison(flipped_cls, right, ctx)

    # Case B & C: exactly one operand is known non-NaN.
    if right.is_known_non_nan:
        if isinstance(node, (Eq, NotEq, Lt, LtEq)):
            return cmp_expr, orig_weight, 0
        return _lower_one_sided_nan_comparison(
            cmp_expr,
            orig_weight,
            nan_side=left,
            non_nan_side=right,
            nan_on_left=True,
            ctx=ctx,
        )
    if left.is_known_non_nan:
        if isinstance(node, (Eq, NotEq, Gt, GtEq)):
            return cmp_expr, orig_weight, 0
        return _lower_one_sided_nan_comparison(
            cmp_expr,
            orig_weight,
            nan_side=right,
            non_nan_side=left,
            nan_on_left=False,
            ctx=ctx,
        )

    # Case D: both operands can be NaN.
    return _lower_two_sided_nan_comparison(cmp_expr, orig_weight, left, right, ctx)


@_type_node.register(IsNan)
@_type_node.register(IsNotNan)
@_type_node.register(IsInfinite)
@_type_node.register(IsFinite)
def _type_float_predicate(
    node: IsNan | IsNotNan | IsInfinite | IsFinite,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    (operand,) = children
    if not _is_exact_typed_operand(operand):
        return _degrade(children, ctx)
    if operand.bq_type is BQType.FLOAT64:
        return _TypedNode(
            node=type(node)(expr=operand.node),
            bq_type=BQType.BOOL,
            time_unit=None,
            is_known_non_nan=True,
            is_known_non_null=operand.is_known_non_null,
            weight=1 + operand.weight,
            has_unsupported=False,
            expansion_cost=operand.expansion_cost,
        )
    if operand.bq_type is BQType.INT64:
        const_value = isinstance(node, (IsNotNan, IsFinite))
        lowered = _build_null_safe_bool_const(
            operand,
            const_value=const_value,
            extra_weight=3,
            ctx=ctx,
        )
        if lowered is None:
            logger.debug(
                "AST node budget exhausted lowering %s on INT64; degrading to Unsupported",
                type(node).__name__,
            )
            return _degrade(children, ctx)
        expr, weight, extra_cost = lowered
        return _TypedNode(
            node=expr,
            bq_type=BQType.BOOL,
            time_unit=None,
            is_known_non_nan=True,
            is_known_non_null=operand.is_known_non_null,
            weight=weight,
            has_unsupported=False,
            expansion_cost=operand.expansion_cost + extra_cost,
        )
    logger.debug(
        "Float predicate %s is not supported for type %r; degrading to Unsupported",
        type(node).__name__,
        operand.bq_type,
    )
    return _degrade(children, ctx)
