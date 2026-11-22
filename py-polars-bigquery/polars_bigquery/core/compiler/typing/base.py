"""Shared type metadata, budget context, unification, and singledispatch registry."""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Mapping, Sequence
from typing import Literal as TypingLiteral

import polars as pl

from polars_bigquery.core.compiler.ir.base import (
    Expr,
    NullLiteral,
    Unsupported,
)
from polars_bigquery.core.compiler.ir.list_ import ListLiteral
from polars_bigquery.core.types import (
    OPAQUE_BQ_TYPES,
    BQType,
)

logger = logging.getLogger(__name__)

MAX_DUPLICATED_OPERAND_NODES = 64

TimeUnitStr = TypingLiteral["ms", "us", "ns"]


@dataclasses.dataclass(frozen=True, slots=True)
class _TypedNode:
    """Inferred type and rewrite metadata for a single lowered IR node."""

    node: Expr
    bq_type: BQType | None
    time_unit: TimeUnitStr | None
    is_known_non_nan: bool
    is_known_non_null: bool
    weight: int
    has_unsupported: bool = False
    is_all_null: bool = False
    decimal_supports_one: bool = True
    expansion_cost: int = 0


_UNSUPPORTED_TYPED_NODE = _TypedNode(
    node=Unsupported(),
    bq_type=None,
    time_unit=None,
    is_known_non_nan=False,
    is_known_non_null=False,
    weight=1,
    has_unsupported=True,
    expansion_cost=0,
)


@dataclasses.dataclass(slots=True)
class _TypingContext:
    """Mutable typing context shared across bottom-up evaluation."""

    schema: pl.Schema
    bq_types: Mapping[str, BQType] | None
    remaining_budget: int

    def consume_budget(self, extra_weight: int) -> bool:
        """Deduct `extra_weight` from the AST expansion budget if sufficient."""
        if extra_weight > self.remaining_budget:
            return False
        self.remaining_budget -= extra_weight
        return True

    def refund_budget(self, cost: int) -> None:
        """Return expansion budget from discarded/degraded child subtrees."""
        if cost > 0:
            self.remaining_budget += cost


def _degrade(
    children: Sequence[_TypedNode],
    ctx: _TypingContext | None = None,
) -> _TypedNode:
    """Degrade an invalid node to bare `Unsupported`, refunding discarded budget."""
    if ctx is not None:
        refunded = sum(child.expansion_cost for child in children)
        if refunded:
            ctx.refund_budget(refunded)
    return _UNSUPPORTED_TYPED_NODE


def _is_exact_typed_operand(child: _TypedNode) -> bool:
    """Return True if `child` is an exact, non-null-literal, non-list typed expression."""
    return (
        child.bq_type is not None
        and not child.has_unsupported
        and not isinstance(child.node, ListLiteral)
    )


def _is_bool_logical_operand(child: _TypedNode) -> bool:
    """Return True if `child` is valid in `And`, `Or`, or `Not`."""
    return (
        child.bq_type is BQType.BOOL
        or child.is_all_null
        or isinstance(child.node, (Unsupported, NullLiteral))
    )


def _is_valid_predicate(pred: _TypedNode) -> bool:
    """Return True if `pred` is valid as a `Ternary` or `When` condition."""
    return pred.bq_type is BQType.BOOL or isinstance(pred.node, Unsupported)


def _is_null_branch(branch: _TypedNode) -> bool:
    """Return True if `branch` is a `NullLiteral` or an all-NULL chained conditional."""
    return isinstance(branch.node, NullLiteral) or branch.is_all_null


def _unify_types(left: BQType, right: BQType) -> BQType | None:
    """Return the BigQuery/Polars-compatible supertype of two scalar types, or None."""
    if left in OPAQUE_BQ_TYPES or right in OPAQUE_BQ_TYPES:
        return None
    if left is right:
        return left
    if (left is BQType.INT64 and right is BQType.FLOAT64) or (
        left is BQType.FLOAT64 and right is BQType.INT64
    ):
        return BQType.FLOAT64
    if (left is BQType.INT64 and right is BQType.NUMERIC) or (
        left is BQType.NUMERIC and right is BQType.INT64
    ):
        return BQType.NUMERIC
    return None


def _unify_many(bq_types: Sequence[BQType]) -> BQType | None:
    """Unify a sequence of BigQuery types, rejecting empty inputs or opaque types."""
    if not bq_types:
        return None
    acc = bq_types[0]
    if acc in OPAQUE_BQ_TYPES:
        return None
    for bq_type in bq_types[1:]:
        unified = _unify_types(acc, bq_type)
        if unified is None:
            return None
        acc = unified
    return acc


@functools.singledispatch
def _type_node(
    node: Expr,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    """Type and lower a single IR node bottom-up; fail closed on unrecognized nodes."""
    logger.debug(
        "Unrecognized IR node %r during typing; degrading to Unsupported",
        node,
    )
    return _degrade(children, ctx)


@_type_node.register
def _type_unsupported(
    _node: Unsupported,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    return _degrade(children, ctx)
