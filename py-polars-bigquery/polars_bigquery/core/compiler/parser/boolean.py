from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

from polars_bigquery.core.compiler.ir.base import Expr, Unsupported
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    Coalesce,
    FillNull,
    IsNotNull,
    IsNull,
    Not,
    Or,
    Ternary,
)
from polars_bigquery.core.compiler.ir.numeric import IntLiteral
from polars_bigquery.core.compiler.parser.base import (
    UNSUPPORTED_RECORD,
    ParseRecord,
    _is_valid_unit_spec,
    extract_function_inputs,
    parse_binary_function,
    parse_binary_op,
    parse_ternary_op,
    parse_unary_function,
    parse_variadic_function,
    register_parser,
)

_FILL_NULL_STRATEGY_VALUES: Mapping[str, int] = {
    "Zero": 0,
    "One": 1,
}


def _construct_fill_null_with_strategy(fill_value: int, *, expr: Expr) -> FillNull:
    """Construct a `FillNull` IR node with an `IntLiteral` replacement value."""
    return FillNull(left=expr, right=IntLiteral(value=fill_value))


def construct_all_horizontal(children: Sequence[Expr]) -> Expr:
    """Fold parsed child `Expr`s into left-associative `And` IR nodes."""
    if not children:
        return Unsupported()
    acc = children[0]
    for child in children[1:]:
        acc = And(left=acc, right=child)
    return acc


@register_parser("$.Literal..Boolean")
def parse_bool_literal(value: Any) -> Expr:
    """Parse a Boolean value from Polars JSON into a BoolLiteral or Unsupported."""
    if not isinstance(value, bool):
        return Unsupported()
    return BoolLiteral(value=value)


@register_parser("$.BinaryExpr.op.And")
def parse_and(op_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `And` BinaryExpr node into an IR `And` record."""
    return parse_binary_op(expr_json, And, op_spec=op_spec)


@register_parser("$.BinaryExpr.op.Or")
def parse_or(op_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Or` BinaryExpr node into an IR `Or` record."""
    return parse_binary_op(expr_json, Or, op_spec=op_spec)


@register_parser("$.Function.function.Boolean.AllHorizontal")
def parse_all_horizontal(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `AllHorizontal` BooleanFunction node into folded `And` IR nodes."""
    if not _is_valid_unit_spec(spec):
        return UNSUPPORTED_RECORD
    inputs = extract_function_inputs(expr_json, require_unit_leaf=True)
    if inputs is None or not inputs:
        return UNSUPPORTED_RECORD
    return ParseRecord("Variadic", construct_all_horizontal, tuple(inputs))


@register_parser("$.Function.function.Boolean.Not")
def parse_not(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Not` BooleanFunction node into an IR `Not` record."""
    return parse_unary_function(expr_json, Not, func_spec=spec)


@register_parser("$.Function.function.Boolean.IsNull")
def parse_is_null(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsNull` BooleanFunction node into an IR `IsNull` record."""
    return parse_unary_function(expr_json, IsNull, func_spec=spec)


@register_parser("$.Function.function.Boolean.IsNotNull")
def parse_is_not_null(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsNotNull` BooleanFunction node into an IR `IsNotNull` record."""
    return parse_unary_function(expr_json, IsNotNull, func_spec=spec)


@register_parser("$.Function.function.Coalesce")
def parse_coalesce(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Coalesce` Function node into an IR `Coalesce` record."""
    return parse_variadic_function(expr_json, Coalesce, func_spec=spec)


@register_parser("$.Function.function.FillNull")
def parse_fill_null(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `FillNull` Function node into an IR `FillNull` record."""
    return parse_binary_function(expr_json, FillNull, func_spec=spec)


@register_parser("$.Function.function.FillNullWithStrategy")
def parse_fill_null_with_strategy(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `FillNullWithStrategy` Function node into an IR `FillNull` record.

    Only scalar constant strategies (`"Zero"` and `"One"`) are supported;
    analytical and aggregate strategies (`"Forward"`, `"Backward"`, `"Min"`,
    `"Max"`, `"Mean"`) cannot be evaluated per-row by the BigQuery Storage Read
    API and return `UNSUPPORTED_RECORD`.
    """
    if not isinstance(spec, str) or spec not in _FILL_NULL_STRATEGY_VALUES:
        return UNSUPPORTED_RECORD
    inputs = extract_function_inputs(expr_json, require_unit_leaf=True)
    if inputs is None or len(inputs) != 1:
        return UNSUPPORTED_RECORD
    return ParseRecord(
        "Unary",
        functools.partial(
            _construct_fill_null_with_strategy,
            _FILL_NULL_STRATEGY_VALUES[spec],
        ),
        (inputs[0],),
    )


@register_parser("$.Ternary")
def parse_ternary(ternary_json: Any) -> ParseRecord:
    """Parse a Polars `Ternary` node into an IR `Ternary` record."""
    return parse_ternary_op(ternary_json, Ternary)
