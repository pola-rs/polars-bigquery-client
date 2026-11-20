from __future__ import annotations

import dataclasses
import functools
import io
import json
from collections.abc import Sequence
from typing import Any

import polars as pl

from polars_bigquery.core.compiler import ir, parser, rewriter, sql
from polars_bigquery.core.compiler.parser import json_to_ir
from polars_bigquery.core.compiler.rewriter import rewrite_ir
from polars_bigquery.core.compiler.sql import ir_to_sql

__all__ = [
    "CompiledPredicate",
    "compile_predicate",
    "format_conjuncts_sql",
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
    """Join SQL conjuncts with left-associative parenthesized `AND`."""
    if not conjuncts:
        return ""
    return functools.reduce(lambda acc, c: f"({acc} AND {c})", conjuncts)


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
    predicate.meta.serialize(predicate_json_file, format="json")
    predicate_json_file.seek(0)
    try:
        predicate_json = json.load(predicate_json_file)
        return _json_expr_to_compiled_predicate(predicate_json)
    except (RecursionError, json.JSONDecodeError):
        return CompiledPredicate(conjuncts=(), is_exact=False)


def predicate_to_row_restriction(predicate: pl.Expr) -> str:
    """Orchestrate compilation of a Polars expression to a BigQuery row restriction SQL string."""
    return compile_predicate(predicate).sql
