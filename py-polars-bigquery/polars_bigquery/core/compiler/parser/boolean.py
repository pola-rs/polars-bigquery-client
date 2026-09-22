from __future__ import annotations

from typing import Any

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
def parse_bool_literal(value: Any) -> BoolLiteral:
    """Parse a Boolean value from Polars JSON into a BoolLiteral."""
    return BoolLiteral(value=bool(value))


@register_parser("$.BinaryExpr.op.And")
def parse_and(_op: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `And` BinaryExpr node into an IR `And` record."""
    return parse_binary_op(expr_json, And)


@register_parser("$.BinaryExpr.op.Or")
def parse_or(_op: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Or` BinaryExpr node into an IR `Or` record."""
    return parse_binary_op(expr_json, Or)


@register_parser("$.Function.function.Boolean.Not")
def parse_not(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Not` BooleanFunction node into an IR `Not` record."""
    return parse_unary_function(expr_json, Not)


@register_parser("$.Function.function.Boolean.IsNull")
def parse_is_null(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsNull` BooleanFunction node into an IR `IsNull` record."""
    return parse_unary_function(expr_json, IsNull)


@register_parser("$.Function.function.Boolean.IsNotNull")
def parse_is_not_null(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `IsNotNull` BooleanFunction node into an IR `IsNotNull` record."""
    return parse_unary_function(expr_json, IsNotNull)
