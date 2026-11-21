from __future__ import annotations

import io
import json
import logging
from collections.abc import Mapping
from typing import Any

import polars as pl
from polars.datatypes import DataType, DataTypeClass

from polars_bigquery.core.compiler import ir, parser, rewriter, sql, typing
from polars_bigquery.core.compiler.ir.base import NullLiteral
from polars_bigquery.core.compiler.parser import json_to_ir
from polars_bigquery.core.compiler.rewriter import rewrite_ir
from polars_bigquery.core.compiler.sql import ir_to_sql
from polars_bigquery.core.compiler.typing import resolve_and_lower_types
from polars_bigquery.core.types import BQType

logger = logging.getLogger(__name__)

MAX_ROW_RESTRICTION_BYTES = 1_000_000

__all__ = [
    "MAX_ROW_RESTRICTION_BYTES",
    "ir",
    "ir_to_sql",
    "json_to_ir",
    "parser",
    "predicate_to_row_restriction",
    "resolve_and_lower_types",
    "rewrite_ir",
    "rewriter",
    "sql",
    "typing",
]


def _utf8_byte_length(sql: str) -> int:
    """Return exact UTF-8 byte length, avoiding buffer allocation for ASCII strings."""
    return len(sql) if sql.isascii() else len(sql.encode("utf-8"))


def _json_expr_to_row_restriction(
    expr_json: dict[str, Any],
    schema: pl.Schema | Mapping[str, DataType | DataTypeClass],
    bq_types: Mapping[str, BQType] | None = None,
) -> str | None:
    """Compile a Polars expression JSON dict into a BigQuery SQL row restriction.

    Returns None if unknown or type-incompatible operators are found and cannot
    guarantee a superset of rows, if the expression reduces to a bare untyped
    `NULL` literal, or if the emitted SQL exceeds `MAX_ROW_RESTRICTION_BYTES`.
    """
    normalized_schema = schema if isinstance(schema, pl.Schema) else pl.Schema(schema)
    ir_tree = json_to_ir(expr_json)
    ir_tree = resolve_and_lower_types(ir_tree, normalized_schema, bq_types=bq_types)
    ir_tree = rewrite_ir(ir_tree)
    # Guard again after boolean-literal folding (e.g. `TRUE AND NULL` -> `NULL`).
    if isinstance(ir_tree, NullLiteral):
        return None
    row_restriction = ir_to_sql(ir_tree)
    if row_restriction is not None:
        byte_len = _utf8_byte_length(row_restriction)
        if byte_len > MAX_ROW_RESTRICTION_BYTES:
            logger.debug(
                "Compiled row_restriction length (%d bytes) exceeds limit (%d bytes); "
                "falling back to client-side filtering",
                byte_len,
                MAX_ROW_RESTRICTION_BYTES,
            )
            return None
    return row_restriction


def predicate_to_row_restriction(
    predicate: pl.Expr,
    *,
    schema: pl.Schema | Mapping[str, DataType | DataTypeClass],
    bq_types: Mapping[str, BQType] | None = None,
) -> str:
    """Orchestrate compilation of a Polars expression to a BigQuery row restriction SQL string."""
    normalized_schema = schema if isinstance(schema, pl.Schema) else pl.Schema(schema)
    predicate_json_file = io.BytesIO()
    try:
        predicate.meta.serialize(predicate_json_file, format="json")
        predicate_json_file.seek(0)
        predicate_json = json.load(predicate_json_file)
        row_restriction = _json_expr_to_row_restriction(
            predicate_json,
            normalized_schema,
            bq_types=bq_types,
        )
    except (RecursionError, json.JSONDecodeError, pl.exceptions.PolarsError):
        logger.debug(
            "Failed to serialize or compile predicate; falling back to client-side filtering",
            exc_info=True,
        )
        row_restriction = None
    return row_restriction if row_restriction is not None else ""
