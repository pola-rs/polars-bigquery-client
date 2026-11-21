"""Typing handlers for comparison operators (`==`, `!=`, `>`, `>=`, `<`, `<=`)."""

from __future__ import annotations

import logging

from polars_bigquery.core.compiler.ir.comparison import (
    Eq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    NotEq,
)
from polars_bigquery.core.compiler.typing.base import (
    _degrade,
    _is_exact_typed_operand,
    _type_node,
    _TypedNode,
    _TypingContext,
    _unify_types,
)
from polars_bigquery.core.compiler.typing.numeric import (
    ComparisonExpr,
    _lower_float_comparison,
)
from polars_bigquery.core.types import BQType

logger = logging.getLogger(__name__)


@_type_node.register(Eq)
@_type_node.register(NotEq)
@_type_node.register(Gt)
@_type_node.register(GtEq)
@_type_node.register(Lt)
@_type_node.register(LtEq)
def _type_comparison(
    node: ComparisonExpr,
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
            "Comparison %s received incompatible or opaque types (%r, %r); degrading to Unsupported",
            type(node).__name__,
            left.bq_type,
            right.bq_type,
        )
        return _degrade(children, ctx)

    if unified is BQType.FLOAT64:
        lowered = _lower_float_comparison(node, left, right, ctx)
        if lowered is None:
            logger.debug(
                "FLOAT64 NaN comparison %s exceeded AST duplication/node budget; degrading to Unsupported",
                type(node).__name__,
            )
            return _degrade(children, ctx)
        expr, weight, extra_cost = lowered
    else:
        expr = type(node)(left=left.node, right=right.node)
        weight = 1 + left.weight + right.weight
        extra_cost = 0

    return _TypedNode(
        node=expr,
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=left.is_known_non_null and right.is_known_non_null,
        weight=weight,
        has_unsupported=False,
        expansion_cost=left.expansion_cost + right.expansion_cost + extra_cost,
    )
