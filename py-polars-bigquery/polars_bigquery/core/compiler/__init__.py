from __future__ import annotations

import dataclasses
import io
import json
import logging
from collections.abc import Sequence
from typing import Any

import polars as pl

from polars_bigquery.core.compiler import ir, parser, rewriter, sql
from polars_bigquery.core.compiler.parser import json_to_ir
from polars_bigquery.core.compiler.rewriter import rewrite_ir
from polars_bigquery.core.compiler.sql import ir_to_sql

logger = logging.getLogger(__name__)

__all__ = [
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
    "rewrite_ir",
    "rewriter",
    "sql",
]


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


def _json_expr_to_compiled_predicate(expr_json: dict[str, Any]) -> CompiledPredicate:
    """Compile a Polars expression JSON dict into structured `AND` conjuncts."""
    ir_tree = json_to_ir(expr_json)
    ir_tree = rewrite_ir(ir_tree)
    conjuncts, is_exact = sql.compile_ir_conjuncts_to_sql(ir_tree)
    return CompiledPredicate(conjuncts=conjuncts, is_exact=is_exact)


def compile_predicate(predicate: pl.Expr) -> CompiledPredicate:
    """Compile a Polars filter expression into structured SQL conjuncts and exactness."""
    predicate_json_file = io.BytesIO()
    try:
        predicate.meta.serialize(predicate_json_file, format="json")
        predicate_json_file.seek(0)
        predicate_json = json.load(predicate_json_file)
        return _json_expr_to_compiled_predicate(predicate_json)
    except (RecursionError, json.JSONDecodeError, pl.exceptions.PolarsError):
        logger.debug(
            "Predicate could not be serialized or parsed; falling back to local filtering.",
        )
        return CompiledPredicate(conjuncts=(), is_exact=False)


def predicate_to_row_restriction(predicate: pl.Expr) -> str:
    """Orchestrate compilation of a Polars expression to a BigQuery row restriction SQL string."""
    return compile_predicate(predicate).sql
