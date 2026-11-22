"""Typing handlers for strict `STRING` IR operators."""

from __future__ import annotations

import logging

from polars_bigquery.core.compiler.ir.base import Expr
from polars_bigquery.core.compiler.ir.string import (
    Contains,
    EndsWith,
    Lowercase,
    StartsWith,
    Uppercase,
)
from polars_bigquery.core.compiler.typing.base import (
    _degrade,
    _is_exact_typed_operand,
    _type_node,
    _TypedNode,
    _TypingContext,
)
from polars_bigquery.core.types import BQType

logger = logging.getLogger(__name__)


@_type_node.register(Uppercase)
@_type_node.register(Lowercase)
def _type_unary_string_op(
    node: Uppercase | Lowercase,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    (operand,) = children
    if not _is_exact_typed_operand(operand) or operand.bq_type is not BQType.STRING:
        logger.debug(
            "String operator %s received non-STRING operand type %r; degrading to Unsupported",
            type(node).__name__,
            operand.bq_type,
        )
        return _degrade(children, ctx)
    return _TypedNode(
        node=type(node)(expr=operand.node),
        bq_type=BQType.STRING,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=operand.is_known_non_null,
        weight=1 + operand.weight,
        has_unsupported=False,
        expansion_cost=operand.expansion_cost,
    )


@_type_node.register(Contains)
@_type_node.register(StartsWith)
@_type_node.register(EndsWith)
def _type_binary_string_op(
    node: Contains | StartsWith | EndsWith,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    left, right = children
    if (
        not _is_exact_typed_operand(left)
        or not _is_exact_typed_operand(right)
        or left.bq_type is not BQType.STRING
        or right.bq_type is not BQType.STRING
    ):
        logger.debug(
            "String predicate %s received non-STRING operand types (%r, %r); degrading to Unsupported",
            type(node).__name__,
            left.bq_type,
            right.bq_type,
        )
        return _degrade(children, ctx)
    reconstructed: Expr = (
        Contains(left=left.node, right=right.node, literal=node.literal)
        if isinstance(node, Contains)
        else type(node)(left=left.node, right=right.node)
    )
    return _TypedNode(
        node=reconstructed,
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=left.is_known_non_null and right.is_known_non_null,
        weight=1 + left.weight + right.weight,
        has_unsupported=False,
        expansion_cost=left.expansion_cost + right.expansion_cost,
    )
