"""Typing handlers for leaf IR nodes: `Column`, `NullLiteral`, `ListLiteral`, and scalar literals."""

from __future__ import annotations

import logging
import math

from polars_bigquery.core.compiler.ir.base import Column, Literal, NullLiteral
from polars_bigquery.core.compiler.ir.boolean import BoolLiteral
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
    _UNSUPPORTED_TYPED_NODE,
    TimeUnitStr,
    _type_node,
    _TypedNode,
    _TypingContext,
)
from polars_bigquery.core.types import (
    BQType,
    polars_datetime_time_unit,
    polars_decimal_supports_one,
    polars_dtype_to_bq_type,
)

logger = logging.getLogger(__name__)


def _scalar_literal_bq_type(node: Literal) -> BQType | None:
    """Infer the non-opaque scalar `BQType` of a scalar literal, rejecting `NaN`."""
    if isinstance(node, BoolLiteral):
        return BQType.BOOL
    if isinstance(node, IntLiteral):
        return BQType.INT64
    if isinstance(node, FloatLiteral):
        return None if math.isnan(node.value) else BQType.FLOAT64
    if isinstance(node, DecimalLiteral):
        return BQType.NUMERIC
    if isinstance(node, StringLiteral):
        return BQType.STRING
    if isinstance(node, BytesLiteral):
        return BQType.BYTES
    if isinstance(node, DateLiteral):
        return BQType.DATE
    if isinstance(node, TimeLiteral):
        return BQType.TIME
    if isinstance(node, TimestampLiteral):
        return BQType.DATETIME if node.tz is None else BQType.TIMESTAMP
    return None


def _list_literal_element_type(node: ListLiteral) -> BQType | None:
    """Return the homogeneous non-opaque, non-NaN scalar element type of a `ListLiteral`."""
    if not node.values:
        return None
    first_type = _scalar_literal_bq_type(node.values[0])
    if first_type is None:
        return None
    for value in node.values[1:]:
        if _scalar_literal_bq_type(value) is not first_type:
            return None
    return first_type


@_type_node.register
def _type_column(
    node: Column,
    _children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    dtype = ctx.schema.get(node.name)
    bq_type: BQType | None = None
    if ctx.bq_types is not None and node.name in ctx.bq_types:
        bq_type = ctx.bq_types[node.name]
    elif dtype is not None:
        bq_type = polars_dtype_to_bq_type(dtype)

    if bq_type is None:
        logger.debug(
            "Column %r has missing or unsupported dtype %r; degrading to Unsupported",
            node.name,
            dtype,
        )
        return _UNSUPPORTED_TYPED_NODE

    time_unit = polars_datetime_time_unit(dtype) if dtype is not None else None
    decimal_supports_one = (
        polars_decimal_supports_one(dtype) if dtype is not None else True
    )
    return _TypedNode(
        node=node,
        bq_type=bq_type,
        time_unit=time_unit,
        is_known_non_nan=bq_type is not BQType.FLOAT64,
        is_known_non_null=False,
        weight=1,
        decimal_supports_one=decimal_supports_one,
    )


@_type_node.register
def _type_null_literal(
    node: NullLiteral,
    _children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    return _TypedNode(
        node=node,
        bq_type=None,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=False,
        weight=1,
    )


@_type_node.register
def _type_list_literal(
    node: ListLiteral,
    _children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    return _TypedNode(
        node=node,
        bq_type=BQType.ARRAY,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=True,
        weight=1 + len(node.values),
    )


@_type_node.register(BoolLiteral)
@_type_node.register(IntLiteral)
@_type_node.register(FloatLiteral)
@_type_node.register(DecimalLiteral)
@_type_node.register(StringLiteral)
@_type_node.register(BytesLiteral)
@_type_node.register(DateLiteral)
@_type_node.register(TimeLiteral)
@_type_node.register(TimestampLiteral)
def _type_scalar_literal(
    node: Literal,
    _children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    if isinstance(node, FloatLiteral):
        return _TypedNode(
            node=node,
            bq_type=BQType.FLOAT64,
            time_unit=None,
            is_known_non_nan=not math.isnan(node.value),
            is_known_non_null=True,
            weight=1,
        )
    if isinstance(node, TimestampLiteral):
        bq_type = BQType.DATETIME if node.tz is None else BQType.TIMESTAMP
        time_unit: TimeUnitStr = (
            "ms" if node.unit == TimestampUnit.MILLISECONDS else "us"
        )
        return _TypedNode(
            node=node,
            bq_type=bq_type,
            time_unit=time_unit,
            is_known_non_nan=True,
            is_known_non_null=True,
            weight=1,
        )
    return _TypedNode(
        node=node,
        bq_type=_scalar_literal_bq_type(node),
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=True,
        weight=1,
    )
