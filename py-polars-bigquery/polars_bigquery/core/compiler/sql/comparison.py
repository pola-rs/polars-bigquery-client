from __future__ import annotations

from polars_bigquery.core.compiler.ir.comparison import (
    Eq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    NotEq,
)
from polars_bigquery.core.compiler.sql.base import format_operator_sql


@format_operator_sql.register
def format_eq(_node: Eq, left: str, right: str) -> str:
    return f"({left} = {right})"


@format_operator_sql.register
def format_not_eq(_node: NotEq, left: str, right: str) -> str:
    return f"({left} != {right})"


@format_operator_sql.register
def format_gt(_node: Gt, left: str, right: str) -> str:
    return f"({left} > {right})"


@format_operator_sql.register
def format_gt_eq(_node: GtEq, left: str, right: str) -> str:
    return f"({left} >= {right})"


@format_operator_sql.register
def format_lt(_node: Lt, left: str, right: str) -> str:
    return f"({left} < {right})"


@format_operator_sql.register
def format_lt_eq(_node: LtEq, left: str, right: str) -> str:
    return f"({left} <= {right})"
