"""Pre-rewrite schema-aware typing and lowering pass for the predicate compiler."""

from __future__ import annotations

import collections
import dataclasses
import functools
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Literal as TypingLiteral

import polars as pl

from polars_bigquery.core.compiler.ir.base import (
    Column,
    Expr,
    Literal,
    NullLiteral,
    Unsupported,
)
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
from polars_bigquery.core.compiler.ir.comparison import (
    Eq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    NotEq,
)
from polars_bigquery.core.compiler.ir.list_ import IsIn, ListLiteral
from polars_bigquery.core.compiler.ir.numeric import (
    DecimalLiteral,
    FloatLiteral,
    IntLiteral,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
)
from polars_bigquery.core.compiler.ir.string import (
    BytesLiteral,
    Contains,
    EndsWith,
    Lowercase,
    StartsWith,
    StringLiteral,
    Uppercase,
)
from polars_bigquery.core.compiler.ir.temporal import (
    DateLiteral,
    TimeLiteral,
    TimestampLiteral,
    TimestampUnit,
)
from polars_bigquery.core.compiler.parser import MAX_AST_NODES
from polars_bigquery.core.types import (
    OPAQUE_BQ_TYPES,
    BQType,
    polars_datetime_time_unit,
    polars_dtype_to_bq_type,
)

logger = logging.getLogger(__name__)

MAX_DUPLICATED_OPERAND_NODES = 64

TimeUnitStr = TypingLiteral["ms", "us", "ns"]
ComparisonExpr = Eq | NotEq | Gt | GtEq | Lt | LtEq

_FLIPPED_COMPARISON_CLS: dict[type[ComparisonExpr], type[ComparisonExpr]] = {
    Eq: Eq,
    NotEq: NotEq,
    Gt: Lt,
    GtEq: LtEq,
    Lt: Gt,
    LtEq: GtEq,
}


@dataclasses.dataclass(frozen=True, slots=True)
class _TypedNode:
    """Inferred type and rewrite metadata for a single lowered IR node."""

    node: Expr
    bq_type: BQType | None
    time_unit: TimeUnitStr | None
    is_known_non_nan: bool
    is_known_non_null: bool
    weight: int
    has_drop_nulls: bool = False
    has_unsupported: bool = False
    is_all_null: bool = False


_UNSUPPORTED_TYPED_NODE = _TypedNode(
    node=Unsupported(),
    bq_type=None,
    time_unit=None,
    is_known_non_nan=False,
    is_known_non_null=False,
    weight=1,
    has_drop_nulls=False,
    has_unsupported=True,
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


def _degrade(children: Sequence[_TypedNode]) -> _TypedNode:
    """Degrade an invalid node to `Unsupported`, preserving valid `DropNulls` siblings."""
    kept_nodes: list[Expr] = []
    kept_weight = 1
    for child in children:
        if child.has_drop_nulls:
            kept_nodes.append(child.node)
            kept_weight += child.weight
    if kept_nodes:
        return _TypedNode(
            node=Unsupported(operands=tuple(kept_nodes)),
            bq_type=None,
            time_unit=None,
            is_known_non_nan=False,
            is_known_non_null=False,
            weight=kept_weight,
            has_drop_nulls=True,
            has_unsupported=True,
        )
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
    return child.bq_type is BQType.BOOL or isinstance(
        child.node, (Unsupported, NullLiteral)
    )


def _is_valid_predicate(pred: _TypedNode) -> bool:
    """Return True if `pred` is valid as a `Ternary` or `When` condition."""
    return pred.bq_type is BQType.BOOL or isinstance(pred.node, Unsupported)


def _unify_types(left: BQType, right: BQType) -> BQType | None:
    """Return the BigQuery/Polars-compatible supertype of two scalar types, or None."""
    if left in OPAQUE_BQ_TYPES or right in OPAQUE_BQ_TYPES:
        return None
    if left is right:
        return left
    pair = {left, right}
    if pair == {BQType.INT64, BQType.FLOAT64}:
        return BQType.FLOAT64
    if pair == {BQType.INT64, BQType.NUMERIC}:
        return BQType.NUMERIC
    return None


def _unify_many(bq_types: Sequence[BQType]) -> BQType | None:
    """Unify a non-empty sequence of BigQuery types, rejecting opaque types."""
    acc = bq_types[0]
    if acc in OPAQUE_BQ_TYPES:
        return None
    for bq_type in bq_types[1:]:
        unified = _unify_types(acc, bq_type)
        if unified is None:
            return None
        acc = unified
    return acc


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


def _build_null_safe_bool_const(
    operand: _TypedNode,
    *,
    const_value: bool,
    extra_weight: int,
    ctx: _TypingContext,
) -> tuple[Expr, int] | None:
    """Build `Ternary(IsNotNull(operand), BoolLiteral(const_value), NullLiteral())`."""
    if not ctx.consume_budget(extra_weight):
        return None
    expr = Ternary(
        predicate=IsNotNull(expr=operand.node),
        truthy=BoolLiteral(value=const_value),
        falsy=NullLiteral(),
    )
    return expr, operand.weight + 4


def _fill_null_strategy_literal(
    bq_type: BQType,
    time_unit: TimeUnitStr | None,
    strategy: str,
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
        return DecimalLiteral(value="1" if is_one else "0")
    if bq_type is BQType.DATE:
        return DateLiteral(days=1 if is_one else 0)
    if bq_type is BQType.STRING:
        return None if is_one else StringLiteral(value="")
    if bq_type is BQType.BYTES:
        return None if is_one else BytesLiteral(value=b"")
    if bq_type is BQType.TIME:
        return None if is_one else TimeLiteral(micros=0)
    # Remaining non-opaque scalar types are DATETIME and TIMESTAMP.
    tz = None if bq_type is BQType.DATETIME else "UTC"
    if not is_one:
        return TimestampLiteral(ticks=0, unit=TimestampUnit.MICROSECONDS, tz=tz)
    if time_unit == "ms":
        return TimestampLiteral(ticks=1, unit=TimestampUnit.MILLISECONDS, tz=tz)
    if time_unit == "us":
        return TimestampLiteral(ticks=1, unit=TimestampUnit.MICROSECONDS, tz=tz)
    return None


def _lower_nan_literal_comparison(
    op_cls: type[ComparisonExpr],
    other: _TypedNode,
    ctx: _TypingContext,
) -> tuple[Expr, int] | None:
    """Lower `other op FloatLiteral(NaN)` (Case A) to exact 3-valued GoogleSQL IR."""
    if op_cls in (Eq, GtEq):
        if other.is_known_non_nan:
            return _build_null_safe_bool_const(
                other, const_value=False, extra_weight=2, ctx=ctx
            )
        return IsNan(expr=other.node), other.weight + 1
    if op_cls in (NotEq, Lt):
        if other.is_known_non_nan:
            return _build_null_safe_bool_const(
                other, const_value=True, extra_weight=2, ctx=ctx
            )
        return Not(expr=IsNan(expr=other.node)), other.weight + 2
    if op_cls is LtEq:
        return _build_null_safe_bool_const(
            other, const_value=True, extra_weight=2, ctx=ctx
        )
    # Remaining comparison operator is Gt.
    return _build_null_safe_bool_const(
        other, const_value=False, extra_weight=2, ctx=ctx
    )


def _lower_one_sided_nan_comparison(
    cmp_expr: ComparisonExpr,
    orig_weight: int,
    *,
    nan_side: _TypedNode,
    non_nan_side: _TypedNode,
    nan_on_left: bool,
    ctx: _TypingContext,
) -> tuple[Expr, int] | None:
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
    return Or(left=cmp_expr, right=nan_guard), orig_weight + extra_weight


def _lower_two_sided_nan_comparison(
    cmp_expr: ComparisonExpr,
    orig_weight: int,
    left: _TypedNode,
    right: _TypedNode,
    ctx: _TypingContext,
) -> tuple[Expr, int] | None:
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
    else:
        # Remaining comparison operator is LtEq.
        extra_weight = 4 + left.weight + right.weight
        lowered = Or(
            left=cmp_expr,
            right=And(left=IsNotNull(expr=left.node), right=IsNan(expr=right.node)),
        )

    if not ctx.consume_budget(extra_weight):
        return None
    return lowered, orig_weight + extra_weight


def _lower_float_comparison(
    node: ComparisonExpr,
    left: _TypedNode,
    right: _TypedNode,
    ctx: _TypingContext,
) -> tuple[Expr, int] | None:
    """Lower a FLOAT64 comparison to preserve Polars' NaN total order under 3VL."""
    orig_weight = 1 + left.weight + right.weight
    cmp_expr: ComparisonExpr = type(node)(left=left.node, right=right.node)

    # Case 0: neither operand can be NaN.
    if left.is_known_non_nan and right.is_known_non_nan:
        return cmp_expr, orig_weight

    # Case A: FloatLiteral(NaN) on either side.
    if isinstance(right.node, FloatLiteral) and math.isnan(right.node.value):
        return _lower_nan_literal_comparison(type(node), left, ctx)
    if isinstance(left.node, FloatLiteral) and math.isnan(left.node.value):
        flipped_cls = _FLIPPED_COMPARISON_CLS[type(node)]
        return _lower_nan_literal_comparison(flipped_cls, right, ctx)

    # Case B & C: exactly one operand is known non-NaN.
    if right.is_known_non_nan:
        if isinstance(node, (Eq, NotEq, Lt, LtEq)):
            return cmp_expr, orig_weight
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
            return cmp_expr, orig_weight
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


def _is_null_branch(branch: _TypedNode) -> bool:
    """Return True if `branch` is a `NullLiteral` or an all-NULL chained conditional."""
    return isinstance(branch.node, NullLiteral) or branch.is_all_null


def _type_conditional(
    node: Ternary | When,
    predicates: Sequence[_TypedNode],
    branches: Sequence[_TypedNode],
    all_children: Sequence[_TypedNode],
    ctx: _TypingContext,
) -> _TypedNode:
    """Shared typing logic for `Ternary` and defensive `When` IR nodes."""
    del ctx
    if not all(_is_valid_predicate(pred) for pred in predicates):
        logger.debug(
            "Conditional %r has non-BOOL or NULL predicate; degrading to Unsupported",
            type(node).__name__,
        )
        return _degrade(all_children)

    if any(
        isinstance(branch.node, ListLiteral) or branch.bq_type in OPAQUE_BQ_TYPES
        for branch in branches
    ):
        logger.debug(
            "Conditional %r has list or opaque value branch; degrading to Unsupported",
            type(node).__name__,
        )
        return _degrade(all_children)

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
    has_drop_nulls = any(child.has_drop_nulls for child in all_children)
    has_unsupported = any(child.has_unsupported for child in all_children)

    if all(_is_null_branch(branch) for branch in branches):
        return _TypedNode(
            node=reconstructed,
            bq_type=None,
            time_unit=None,
            is_known_non_nan=True,
            is_known_non_null=False,
            weight=weight,
            has_drop_nulls=has_drop_nulls,
            has_unsupported=has_unsupported,
            is_all_null=True,
        )

    if any(isinstance(branch.node, Unsupported) for branch in branches):
        known_branches = [
            branch
            for branch in branches
            if not _is_null_branch(branch) and not isinstance(branch.node, Unsupported)
        ]
        if all(branch.bq_type is BQType.BOOL for branch in known_branches):
            return _TypedNode(
                node=reconstructed,
                bq_type=BQType.BOOL,
                time_unit=None,
                is_known_non_nan=True,
                is_known_non_null=False,
                weight=weight,
                has_drop_nulls=has_drop_nulls,
                has_unsupported=True,
            )
        logger.debug(
            "Conditional %r with Unsupported value branch has non-BOOL sibling branch; degrading to Unsupported",
            type(node).__name__,
        )
        return _degrade(all_children)

    non_null_branches = [branch for branch in branches if not _is_null_branch(branch)]
    unified = _unify_many(
        [branch.bq_type for branch in non_null_branches if branch.bq_type is not None]
    )
    if unified is None:
        logger.debug(
            "Conditional %r value branches have incompatible types; degrading to Unsupported",
            type(node).__name__,
        )
        return _degrade(all_children)

    time_units = {branch.time_unit for branch in non_null_branches}
    time_unit = next(iter(time_units)) if len(time_units) == 1 else None
    is_known_non_nan = all(branch.is_known_non_nan for branch in non_null_branches)
    is_known_non_null = (
        not has_unsupported
        and all(pred.is_known_non_null for pred in predicates)
        and all(branch.is_known_non_null for branch in branches)
    )
    return _TypedNode(
        node=reconstructed,
        bq_type=unified,
        time_unit=time_unit,
        is_known_non_nan=is_known_non_nan,
        is_known_non_null=is_known_non_null,
        weight=weight,
        has_drop_nulls=has_drop_nulls,
        has_unsupported=has_unsupported,
    )


@functools.singledispatch
def _type_node(
    node: Expr,
    children: tuple[_TypedNode, ...],
    ctx: _TypingContext,
) -> _TypedNode:
    """Type and lower a single IR node bottom-up; fail closed on unrecognized nodes."""
    del ctx
    logger.debug(
        "Unrecognized IR node %r during typing; degrading to Unsupported",
        node,
    )
    return _degrade(children)


@_type_node.register
def _type_unsupported(
    _node: Unsupported,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    return _degrade(children)


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
    return _TypedNode(
        node=node,
        bq_type=bq_type,
        time_unit=time_unit,
        is_known_non_nan=bq_type is not BQType.FLOAT64,
        is_known_non_null=False,
        weight=1,
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


@_type_node.register(And)
@_type_node.register(Or)
def _type_binary_logical(
    node: And | Or,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    left, right = children
    if not (_is_bool_logical_operand(left) and _is_bool_logical_operand(right)):
        logger.debug(
            "Logical operator %s received non-BOOL operand types (%r, %r); degrading to Unsupported",
            type(node).__name__,
            left.bq_type,
            right.bq_type,
        )
        return _degrade(children)
    has_unsupported = left.has_unsupported or right.has_unsupported
    return _TypedNode(
        node=type(node)(left=left.node, right=right.node),
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=(
            not has_unsupported and left.is_known_non_null and right.is_known_non_null
        ),
        weight=1 + left.weight + right.weight,
        has_drop_nulls=left.has_drop_nulls or right.has_drop_nulls,
        has_unsupported=has_unsupported,
    )


@_type_node.register
def _type_not(
    _node: Not,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    (operand,) = children
    if not _is_bool_logical_operand(operand):
        logger.debug(
            "Logical NOT received non-BOOL operand type %r; degrading to Unsupported",
            operand.bq_type,
        )
        return _degrade(children)
    return _TypedNode(
        node=Not(expr=operand.node),
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=not operand.has_unsupported and operand.is_known_non_null,
        weight=1 + operand.weight,
        has_drop_nulls=operand.has_drop_nulls,
        has_unsupported=operand.has_unsupported,
    )


@_type_node.register(IsNull)
@_type_node.register(IsNotNull)
def _type_null_check(
    node: IsNull | IsNotNull,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    (operand,) = children
    if not _is_exact_typed_operand(operand):
        logger.debug(
            "%s received invalid operand %r; degrading to Unsupported",
            type(node).__name__,
            operand.node,
        )
        return _degrade(children)
    return _TypedNode(
        node=type(node)(expr=operand.node),
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=True,
        weight=1 + operand.weight,
        has_drop_nulls=operand.has_drop_nulls,
        has_unsupported=False,
    )


@_type_node.register
def _type_drop_nulls(
    _node: DropNulls,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    (operand,) = children
    if not _is_exact_typed_operand(operand):
        logger.debug(
            "DropNulls received invalid operand %r; degrading to bare Unsupported",
            operand.node,
        )
        return _UNSUPPORTED_TYPED_NODE
    return _TypedNode(
        node=DropNulls(expr=operand.node),
        bq_type=operand.bq_type,
        time_unit=operand.time_unit,
        is_known_non_nan=operand.is_known_non_nan,
        is_known_non_null=False,
        weight=1 + operand.weight,
        has_drop_nulls=True,
        has_unsupported=False,
    )


@_type_node.register(Uppercase)
@_type_node.register(Lowercase)
def _type_unary_string_op(
    node: Uppercase | Lowercase,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    (operand,) = children
    if not _is_exact_typed_operand(operand) or operand.bq_type is not BQType.STRING:
        logger.debug(
            "String operator %s received non-STRING operand type %r; degrading to Unsupported",
            type(node).__name__,
            operand.bq_type,
        )
        return _degrade(children)
    return _TypedNode(
        node=type(node)(expr=operand.node),
        bq_type=BQType.STRING,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=operand.is_known_non_null,
        weight=1 + operand.weight,
        has_drop_nulls=operand.has_drop_nulls,
        has_unsupported=False,
    )


@_type_node.register(Contains)
@_type_node.register(StartsWith)
@_type_node.register(EndsWith)
def _type_binary_string_op(
    node: Contains | StartsWith | EndsWith,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
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
        return _degrade(children)
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
        has_drop_nulls=left.has_drop_nulls or right.has_drop_nulls,
        has_unsupported=False,
    )


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
        return _degrade(children)
    if operand.bq_type is BQType.FLOAT64:
        return _TypedNode(
            node=type(node)(expr=operand.node),
            bq_type=BQType.BOOL,
            time_unit=None,
            is_known_non_nan=True,
            is_known_non_null=operand.is_known_non_null,
            weight=1 + operand.weight,
            has_drop_nulls=operand.has_drop_nulls,
            has_unsupported=False,
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
            return _degrade(children)
        expr, weight = lowered
        return _TypedNode(
            node=expr,
            bq_type=BQType.BOOL,
            time_unit=None,
            is_known_non_nan=True,
            is_known_non_null=operand.is_known_non_null,
            weight=weight,
            has_drop_nulls=operand.has_drop_nulls,
            has_unsupported=False,
        )
    logger.debug(
        "Float predicate %s is not supported for type %r; degrading to Unsupported",
        type(node).__name__,
        operand.bq_type,
    )
    return _degrade(children)


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
        return _degrade(children)

    fill_literal = _fill_null_strategy_literal(
        operand.bq_type,
        operand.time_unit,
        node.strategy,
    )
    if fill_literal is None:
        logger.debug(
            "FillNullWithStrategy(%r) is not representable for type %r (time_unit=%r); degrading to Unsupported",
            node.strategy,
            operand.bq_type,
            operand.time_unit,
        )
        return _degrade(children)

    if not ctx.consume_budget(1):
        logger.debug(
            "AST node budget exhausted lowering FillNullWithStrategy(%r); degrading to Unsupported",
            node.strategy,
        )
        return _degrade(children)

    return _TypedNode(
        node=FillNull(left=operand.node, right=fill_literal),
        bq_type=operand.bq_type,
        time_unit=operand.time_unit,
        is_known_non_nan=operand.is_known_non_nan,
        is_known_non_null=True,
        weight=operand.weight + 2,
        has_drop_nulls=operand.has_drop_nulls,
        has_unsupported=False,
    )


@_type_node.register
def _type_fill_null(
    _node: FillNull,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    left, right = children
    if (
        not _is_exact_typed_operand(left)
        or not _is_exact_typed_operand(right)
        or left.bq_type is None
        or right.bq_type is None
    ):
        return _degrade(children)
    unified = _unify_types(left.bq_type, right.bq_type)
    if unified is None:
        logger.debug(
            "FillNull operands have incompatible types (%r, %r); degrading to Unsupported",
            left.bq_type,
            right.bq_type,
        )
        return _degrade(children)
    time_unit = left.time_unit if left.time_unit == right.time_unit else None
    return _TypedNode(
        node=FillNull(left=left.node, right=right.node),
        bq_type=unified,
        time_unit=time_unit,
        is_known_non_nan=left.is_known_non_nan and right.is_known_non_nan,
        is_known_non_null=left.is_known_non_null or right.is_known_non_null,
        weight=1 + left.weight + right.weight,
        has_drop_nulls=left.has_drop_nulls or right.has_drop_nulls,
        has_unsupported=False,
    )


@_type_node.register
def _type_coalesce(
    _node: Coalesce,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
) -> _TypedNode:
    if not children or any(not _is_exact_typed_operand(child) for child in children):
        return _degrade(children)
    unified = _unify_many(
        [child.bq_type for child in children if child.bq_type is not None]
    )
    if unified is None:
        logger.debug(
            "Coalesce operands have incompatible or opaque types; degrading to Unsupported"
        )
        return _degrade(children)
    time_units = {child.time_unit for child in children}
    time_unit = next(iter(time_units)) if len(time_units) == 1 else None
    return _TypedNode(
        node=Coalesce(operands=tuple(child.node for child in children)),
        bq_type=unified,
        time_unit=time_unit,
        is_known_non_nan=all(child.is_known_non_nan for child in children),
        is_known_non_null=any(child.is_known_non_null for child in children),
        weight=1 + sum(child.weight for child in children),
        has_drop_nulls=any(child.has_drop_nulls for child in children),
        has_unsupported=False,
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
        return _degrade(children)
    predicates = children[:-1:2]
    branches = (*children[1::2], children[-1])
    return _type_conditional(node, predicates, branches, children, ctx)


@_type_node.register
def _type_is_in(
    _node: IsIn,
    children: tuple[_TypedNode, ...],
    _ctx: _TypingContext,
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
        return _degrade(children)
    elem_type = _list_literal_element_type(right.node)
    if elem_type is not left.bq_type:
        logger.debug(
            "IsIn operand type %r does not strictly match list element type %r; degrading to Unsupported",
            left.bq_type,
            elem_type,
        )
        return _degrade(children)
    return _TypedNode(
        node=IsIn(left=left.node, right=right.node),
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=left.is_known_non_null,
        weight=1 + left.weight + right.weight,
        has_drop_nulls=left.has_drop_nulls,
        has_unsupported=False,
    )


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
        return _degrade(children)
    unified = _unify_types(left.bq_type, right.bq_type)
    if unified is None:
        logger.debug(
            "Comparison %s received incompatible or opaque types (%r, %r); degrading to Unsupported",
            type(node).__name__,
            left.bq_type,
            right.bq_type,
        )
        return _degrade(children)

    if unified is BQType.FLOAT64:
        lowered = _lower_float_comparison(node, left, right, ctx)
        if lowered is None:
            logger.debug(
                "FLOAT64 NaN comparison %s exceeded AST duplication/node budget; degrading to Unsupported",
                type(node).__name__,
            )
            return _degrade(children)
        expr, weight = lowered
    else:
        expr = type(node)(left=left.node, right=right.node)
        weight = 1 + left.weight + right.weight

    return _TypedNode(
        node=expr,
        bq_type=BQType.BOOL,
        time_unit=None,
        is_known_non_nan=True,
        is_known_non_null=left.is_known_non_null and right.is_known_non_null,
        weight=weight,
        has_drop_nulls=left.has_drop_nulls or right.has_drop_nulls,
        has_unsupported=False,
    )


def resolve_and_lower_types(
    root: Expr,
    schema: pl.Schema,
    bq_types: Mapping[str, BQType] | None = None,
) -> Expr:
    """Resolve column/expression types and lower schema-dependent IR nodes iteratively.

    Performs a two-pass iterative traversal (zero Python recursion):
    1. Pass 1 (Top-down BFS indexing): Flattens `root` into breadth-first order and
       computes total input AST node weight (counting `ListLiteral` as `1 + len(values)`).
    2. Pass 2 (Bottom-up reverse-BFS evaluation): Evaluates types, enforces operator
       compatibility, lowers `FillNullWithStrategy`, `INT64` float predicates, and
       `FLOAT64` `NaN` comparisons within `MAX_AST_NODES`, and validates the root type.
    """
    nodes: list[Expr] = [root]
    children_map: list[tuple[int, ...]] = [()]
    queue: collections.deque[int] = collections.deque([0])
    total_input_weight = 0

    while queue:
        curr_idx = queue.popleft()
        node = nodes[curr_idx]
        total_input_weight += (
            1 + len(node.values) if isinstance(node, ListLiteral) else 1
        )
        node_children = node.children()
        if node_children:
            child_indices: list[int] = []
            for child in node_children:
                c_idx = len(nodes)
                nodes.append(child)
                children_map.append(())
                queue.append(c_idx)
                child_indices.append(c_idx)
            children_map[curr_idx] = tuple(child_indices)

    ctx = _TypingContext(
        schema=schema,
        bq_types=bq_types,
        remaining_budget=max(0, MAX_AST_NODES - total_input_weight),
    )

    typed_nodes: list[_TypedNode] = [_UNSUPPORTED_TYPED_NODE] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        child_typed = tuple(typed_nodes[c_idx] for c_idx in children_map[idx])
        typed_nodes[idx] = _type_node(nodes[idx], child_typed, ctx)

    root_typed = typed_nodes[0]
    if root_typed.is_all_null or (
        root_typed.bq_type is not None and root_typed.bq_type is not BQType.BOOL
    ):
        logger.debug(
            "Root predicate has non-BOOL or all-NULL type %r; degrading to Unsupported",
            root_typed.bq_type,
        )
        return _degrade((root_typed,)).node

    return root_typed.node


__all__ = [
    "MAX_DUPLICATED_OPERAND_NODES",
    "resolve_and_lower_types",
]
