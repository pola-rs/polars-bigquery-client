from __future__ import annotations

from typing import Any

from polars_bigquery.core.compiler.ir.comparison import (
    Eq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    NotEq,
)
from polars_bigquery.core.compiler.parser.base import (
    ParseRecord,
    parse_binary_op,
    register_parser,
)


@register_parser("$.BinaryExpr.op.Eq")
def parse_eq(_op: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Eq` BinaryExpr node into an IR `Eq` record."""
    return parse_binary_op(expr_json, Eq)


@register_parser("$.BinaryExpr.op.NotEq")
def parse_not_eq(_op: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `NotEq` BinaryExpr node into an IR `NotEq` record."""
    return parse_binary_op(expr_json, NotEq)


@register_parser("$.BinaryExpr.op.Gt")
def parse_gt(_op: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Gt` BinaryExpr node into an IR `Gt` record."""
    return parse_binary_op(expr_json, Gt)


@register_parser("$.BinaryExpr.op.GtEq")
def parse_gt_eq(_op: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `GtEq` BinaryExpr node into an IR `GtEq` record."""
    return parse_binary_op(expr_json, GtEq)


@register_parser("$.BinaryExpr.op.Lt")
def parse_lt(_op: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Lt` BinaryExpr node into an IR `Lt` record."""
    return parse_binary_op(expr_json, Lt)


@register_parser("$.BinaryExpr.op.LtEq")
def parse_lt_eq(_op: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `LtEq` BinaryExpr node into an IR `LtEq` record."""
    return parse_binary_op(expr_json, LtEq)
