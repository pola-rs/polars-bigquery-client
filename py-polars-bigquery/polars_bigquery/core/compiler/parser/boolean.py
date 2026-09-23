from __future__ import annotations

from typing import Any

from polars_bigquery.core.compiler.ir.base import Expr, Unsupported
from polars_bigquery.core.compiler.ir.boolean import (
    And,
    BoolLiteral,
    IsNotNull,
    IsNull,
    Not,
    Or,
)
from polars_bigquery.core.compiler.parser.base import (
    ParseRecord,
    parse_binary_op,
    parse_unary_function,
    register_parser,
)


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
