from __future__ import annotations

import io
import json
from typing import Any

import polars as pl

from polars_bigquery.core.predicates import ir, sql
from polars_bigquery.core.predicates.ir import json_to_ir
from polars_bigquery.core.predicates.sql import (
    _column_to_sql_identifier,
    _escape_sql_string,
    _is_allowed_unicode_category,
    _is_valid_column_character,
    _json_literal_to_sql,
    ir_to_sql,
)

__all__ = [
    "_column_to_sql_identifier",
    "_escape_sql_string",
    "_is_allowed_unicode_category",
    "_is_valid_column_character",
    "_json_expr_to_row_restriction",
    "_json_literal_to_sql",
    "ir",
    "ir_to_sql",
    "json_to_ir",
    "predicate_to_row_restriction",
    "sql",
]


def _json_expr_to_row_restriction(expr_json: dict[str, Any]) -> str | None:
    """Compile a Polars expression JSON dict into a BigQuery SQL row restriction.

    Returns None if unknown operators are found and can't guarantee a superset of rows.
    """
    ir_tree = json_to_ir(expr_json)
    return ir_to_sql(ir_tree)


def predicate_to_row_restriction(predicate: pl.Expr) -> str:
    """Orchestrate compilation of a Polars expression to a BigQuery row restriction SQL string."""
    predicate_json_file = io.BytesIO()
    predicate.meta.serialize(predicate_json_file, format="json")
    predicate_json_file.seek(0)
    try:
        predicate_json = json.load(predicate_json_file)
        row_restriction = _json_expr_to_row_restriction(predicate_json)
    except RecursionError:
        row_restriction = None
    return row_restriction if row_restriction is not None else ""
