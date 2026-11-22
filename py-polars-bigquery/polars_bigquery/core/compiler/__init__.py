from __future__ import annotations

import dataclasses
import io
import json
import logging
from collections.abc import Mapping, Sequence
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
    "CompiledPredicate",
    "compile_predicate",
    "format_conjuncts_sql",
    "formatted_conjuncts_byte_length",
    "formatted_conjuncts_byte_length_from_stats",
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


def _utf8_byte_length(text: str) -> int:
    """Return exact UTF-8 byte length, avoiding buffer allocation for ASCII strings."""
    return len(text) if text.isascii() else len(text.encode("utf-8"))


def format_conjuncts_sql(conjuncts: Sequence[str]) -> str:
    """Join SQL conjuncts with flat, single-pass parenthesized `AND`.

    Emits `""` for 0 conjuncts, `conjuncts[0]` for 1 conjunct, and
    `({c0} AND {c1} AND ...)` for 2+ conjuncts using a single `O(N)` join so
    large conjunctions avoid `O(N^2)` string reallocation and `(N-1)`-deep
    parenthesis nesting in BigQuery's SQL parser.

    Precondition: each item in `conjuncts` MUST be an atomic or already
    parenthesized SQL boolean expression that binds at least as tightly as
    `AND` (for example, conjuncts produced by `compile_ir_conjuncts_to_sql`).
    Passing an unparenthesized `OR` expression (such as ``"a = 1 OR b = 2"``)
    would change operator precedence relative to `AND`.
    """
    if not conjuncts:
        return ""
    if len(conjuncts) == 1:
        return conjuncts[0]
    return f"({' AND '.join(conjuncts)})"


def formatted_conjuncts_byte_length_from_stats(count: int, total_bytes: int) -> int:
    """Return the exact UTF-8 byte length of `format_conjuncts_sql(...)` from `(count, total_bytes)`."""
    if count <= 0:
        return 0
    if count == 1:
        return total_bytes
    # "(" + c0 + " AND " + c1 + ... + ")"  -> 5 bytes per " AND " + 2 outer parens
    return total_bytes + 5 * (count - 1) + 2


def formatted_conjuncts_byte_length(conjunct_byte_lengths: Sequence[int]) -> int:
    """Return the exact UTF-8 byte length of `format_conjuncts_sql(...)` in `O(1)` memory."""
    return formatted_conjuncts_byte_length_from_stats(
        len(conjunct_byte_lengths),
        sum(conjunct_byte_lengths),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CompiledPredicate:
    """Structured result of compiling a Polars filter expression into SQL."""

    conjuncts: tuple[str, ...]
    is_exact: bool

    @property
    def sql(self) -> str:
        """Combined BigQuery SQL boolean expression for all conjuncts."""
        return format_conjuncts_sql(self.conjuncts)


def _json_expr_to_compiled_predicate(
    expr_json: dict[str, Any],
    schema: pl.Schema | Mapping[str, DataType | DataTypeClass],
    bq_types: Mapping[str, BQType] | None = None,
) -> CompiledPredicate:
    """Compile a Polars expression JSON dict into structured `AND` conjuncts."""
    normalized_schema = schema if isinstance(schema, pl.Schema) else pl.Schema(schema)
    ir_tree = json_to_ir(expr_json)
    ir_tree = resolve_and_lower_types(ir_tree, normalized_schema, bq_types=bq_types)
    ir_tree = rewrite_ir(ir_tree)
    # Guard again after boolean-literal folding (e.g. `TRUE AND NULL` -> `NULL`).
    if isinstance(ir_tree, NullLiteral):
        return CompiledPredicate(conjuncts=(), is_exact=False)
    conjuncts, is_exact = sql.compile_ir_conjuncts_to_sql(ir_tree)
    return CompiledPredicate(conjuncts=conjuncts, is_exact=is_exact)


def compile_predicate(
    predicate: pl.Expr,
    *,
    schema: pl.Schema | Mapping[str, DataType | DataTypeClass],
    bq_types: Mapping[str, BQType] | None = None,
) -> CompiledPredicate:
    """Compile a Polars filter expression into structured SQL conjuncts and exactness."""
    normalized_schema = schema if isinstance(schema, pl.Schema) else pl.Schema(schema)
    predicate_json_file = io.BytesIO()
    try:
        predicate.meta.serialize(predicate_json_file, format="json")
        predicate_json_file.seek(0)
        predicate_json = json.load(predicate_json_file)
        return _json_expr_to_compiled_predicate(
            predicate_json,
            normalized_schema,
            bq_types=bq_types,
        )
    except (RecursionError, json.JSONDecodeError, pl.exceptions.PolarsError):
        logger.debug(
            "Predicate could not be serialized or parsed; falling back to local filtering.",
            exc_info=True,
        )
        return CompiledPredicate(conjuncts=(), is_exact=False)


def predicate_to_row_restriction(
    predicate: pl.Expr,
    *,
    schema: pl.Schema | Mapping[str, DataType | DataTypeClass],
    bq_types: Mapping[str, BQType] | None = None,
) -> str:
    """Orchestrate compilation of a Polars expression to a BigQuery row restriction SQL string."""
    row_restriction = compile_predicate(
        predicate,
        schema=schema,
        bq_types=bq_types,
    ).sql
    if row_restriction:
        byte_len = _utf8_byte_length(row_restriction)
        if byte_len > MAX_ROW_RESTRICTION_BYTES:
            logger.debug(
                "Compiled row_restriction length (%d bytes) exceeds limit (%d bytes); "
                "falling back to client-side filtering",
                byte_len,
                MAX_ROW_RESTRICTION_BYTES,
            )
            return ""
    return row_restriction
