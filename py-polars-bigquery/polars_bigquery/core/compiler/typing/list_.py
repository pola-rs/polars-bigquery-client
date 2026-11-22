"""Typing handler for set-membership (`IsIn`) IR nodes."""

from __future__ import annotations

import logging

from polars_bigquery.core.compiler.ir.list_ import IsIn, ListLiteral
from polars_bigquery.core.compiler.typing.base import (
    _degrade,
    _is_exact_typed_operand,
    _type_node,
    _TypedNode,
    _TypingContext,
)
from polars_bigquery.core.compiler.typing.literals import _list_literal_element_type
from polars_bigquery.core.types import OPAQUE_BQ_TYPES, BQType

logger = logging.getLogger(__name__)


@_type_node.register
def _type_is_in(
    _node: IsIn,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    left, right = children
    if (
        not _is_exact_typed_operand(left)
        or left.bq_type in OPAQUE_BQ_TYPES
        or not isinstance(right.node, ListLiteral)
    ):
        logger.debug(
            "IsIn received invalid operands (%r, %r); degrading to Unsupported",
            left.bq_type,
            type(right.node).__name__,
        )
        return _degrade(children, ctx)
    elem_type = _list_literal_element_type(right.node)
    if elem_type is not left.bq_type:
        logger.debug(
            "IsIn operand type %r does not strictly match list element type %r; degrading to Unsupported",
            left.bq_type,
            elem_type,
        )
        return _degrade(children, ctx)
    return _TypedNode(
        node=IsIn(left=left.node, right=right.node),
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=left.is_known_non_null,
        weight=1 + left.weight + right.weight,
        has_unsupported=False,
        expansion_cost=left.expansion_cost,
    )
