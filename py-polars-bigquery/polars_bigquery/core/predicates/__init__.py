from __future__ import annotations

import functools
import io
import json
import operator
from collections import deque
from dataclasses import dataclass
from typing import Any

import polars as pl

import polars_bigquery.exceptions
from polars_bigquery.core.predicates import ir, sql
from polars_bigquery.core.predicates.ir import (
    Expr,
    Literal,
    Unsupported,
    _json_literal_to_ir,
    json_to_ir,
)
from polars_bigquery.core.predicates.sql import (
    _column_to_sql_identifier,
    _escape_sql_string,
    _is_allowed_unicode_category,
    _is_valid_column_character,
    _literal_ir_to_sql,
    ir_to_sql,
)

__all__ = [
    "CompiledPredicate",
    "_column_to_sql_identifier",
    "_escape_sql_string",
    "_is_allowed_unicode_category",
    "_is_valid_column_character",
    "_json_expr_to_row_restriction",
    "_json_literal_to_sql",
    "compile_predicate",
    "ir",
    "ir_to_sql",
    "json_to_ir",
    "predicate_to_row_restriction",
    "sql",
]


@dataclass(frozen=True)
class CompiledPredicate:
    """Result of compiling a Polars predicate expression for BigQuery scan pushdown."""

    row_restriction: str
    residual_predicate: pl.Expr | None
    referenced_columns: tuple[str, ...]


def _json_literal_to_sql(literal_json: dict[str, Any]) -> str | None:
    """Convert a literal from a Polars expression JSON into SQL-like."""
    ir_node = _json_literal_to_ir(literal_json)
    if isinstance(ir_node, Literal):
        return _literal_ir_to_sql(ir_node)
    return None


def _is_ir_exact(root: Expr) -> bool:
    """Return True if the IR tree contains no Unsupported nodes."""
    queue: deque[Expr] = deque([root])
    while queue:
        node = queue.popleft()
        if isinstance(node, Unsupported):
            return False
        queue.extend(node.children())
    return True


def _split_and_conjuncts_json(expr_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract top-level AND conjunct dicts from a Polars serialized expression JSON."""
    conjuncts: list[dict[str, Any]] = []
    stack = [expr_json]
    while stack:
        curr = stack.pop()
        if (
            isinstance(curr, dict)
            and "BinaryExpr" in curr
            and isinstance(curr["BinaryExpr"], dict)
            and curr["BinaryExpr"].get("op") == "And"
        ):
            right = curr["BinaryExpr"].get("right")
            left = curr["BinaryExpr"].get("left")
            if right is not None:
                stack.append(right)
            if left is not None:
                stack.append(left)
        else:
            conjuncts.append(curr)
    return conjuncts


def _extract_json_columns(expr_json: Any) -> set[str]:
    """Extract all column names referenced in a Polars serialized expression JSON dict."""
    cols: set[str] = set()
    stack = [expr_json]
    while stack:
        curr = stack.pop()
        if isinstance(curr, dict):
            if "Column" in curr and isinstance(curr["Column"], str):
                cols.add(curr["Column"])
            else:
                stack.extend(curr.values())
        elif isinstance(curr, list):
            stack.extend(curr)
    return cols


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
    except (RecursionError, json.JSONDecodeError):
        row_restriction = None
    return row_restriction if row_restriction is not None else ""


def compile_predicate(
    predicate: pl.Expr,
    pseudo_columns: tuple[str, ...] = (),
) -> CompiledPredicate:
    """Compile a Polars predicate into a BigQuery row_restriction and client-side residual filter.

    Enforces strict pushdown safety when pseudo-columns (e.g., _PARTITIONDATE, _PARTITIONTIME)
    are referenced:
    - Any top-level AND conjunct referencing a pseudo-column MUST be 100% pushable to BigQuery SQL.
      If it cannot be pushed down, BigQueryError is raised to prevent silent data corruption
      (since pseudo-columns are synthesized as NULL client-side).
    - Top-level AND conjuncts referencing only physical columns are pushed down if supported AND
      retained in residual_predicate so Polars filters physical columns client-side.
    """
    root_names = tuple(dict.fromkeys(predicate.meta.root_names()))
    if set(root_names).isdisjoint(pseudo_columns):
        return CompiledPredicate(
            row_restriction=predicate_to_row_restriction(predicate),
            residual_predicate=predicate,
            referenced_columns=root_names,
        )

    predicate_json_file = io.BytesIO()
    predicate.meta.serialize(predicate_json_file, format="json")
    predicate_json_file.seek(0)
    try:
        predicate_json = json.load(predicate_json_file)
    except (RecursionError, json.JSONDecodeError) as exc:
        msg = (
            "Failed to serialize predicate referencing BigQuery pseudo-column(s) "
            f"{sorted(set(root_names).intersection(pseudo_columns))}."
        )
        raise polars_bigquery.exceptions.BigQueryError(msg) from exc

    conjunct_jsons = _split_and_conjuncts_json(predicate_json)
    sql_conjuncts: list[str] = []
    residual_exprs: list[pl.Expr] = []
    residual_cols: set[str] = set()

    for c_json in conjunct_jsons:
        c_cols = _extract_json_columns(c_json)
        c_ir = json_to_ir(c_json)
        c_sql, is_exact = sql._compile_ir_to_sql(c_ir)

        if not c_cols.isdisjoint(pseudo_columns):
            if not is_exact or c_sql is None:
                bad_cols = sorted(c_cols.intersection(pseudo_columns))
                msg = (
                    f"Predicate referencing BigQuery pseudo-column(s) {bad_cols} "
                    "cannot be fully pushed down to BigQuery Storage Read API SQL. "
                    "In-memory evaluation is not possible because pseudo-columns are "
                    "synthesized as NULL."
                )
                raise polars_bigquery.exceptions.BigQueryError(msg)
            sql_conjuncts.append(c_sql)
        else:
            if c_sql is not None:
                sql_conjuncts.append(c_sql)
            c_expr = pl.Expr.deserialize(
                io.BytesIO(json.dumps(c_json).encode()), format="json"
            )
            residual_exprs.append(c_expr)
            residual_cols.update(c_cols)

    if not sql_conjuncts:
        row_restriction = ""
    elif len(sql_conjuncts) == 1:
        row_restriction = sql_conjuncts[0]
    else:
        row_restriction = functools.reduce(
            lambda left, right: f"({left} AND {right})", sql_conjuncts
        )

    residual_predicate = (
        functools.reduce(operator.and_, residual_exprs) if residual_exprs else None
    )
    referenced_columns = tuple(col for col in root_names if col in residual_cols)

    return CompiledPredicate(
        row_restriction=row_restriction,
        residual_predicate=residual_predicate,
        referenced_columns=referenced_columns,
    )
