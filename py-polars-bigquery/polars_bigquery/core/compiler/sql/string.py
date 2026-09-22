from __future__ import annotations

from polars_bigquery.core.compiler.ir.string import (
    Contains,
    EndsWith,
    Lowercase,
    StartsWith,
    StringLiteral,
    Uppercase,
)
from polars_bigquery.core.compiler.sql.base import (
    format_operator_sql,
    literal_ir_to_sql,
)


def escape_sql_string(val: str) -> str:
    """Escape a Python string into a single-quoted BigQuery SQL string literal."""
    escaped = (
        val.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\0", "\\x00")
    )
    return f"'{escaped}'"


_escape_sql_string = escape_sql_string


@literal_ir_to_sql.register
def format_string_literal(node: StringLiteral) -> str:
    """Format a StringLiteral IR node as a BigQuery SQL string literal."""
    return escape_sql_string(node.value)


@format_operator_sql.register
def format_contains(node: Contains, left: str, right: str) -> str:
    if node.literal:
        return f"(STRPOS({left}, {right}) > 0)"
    return f"REGEXP_CONTAINS({left}, {right})"


@format_operator_sql.register
def format_starts_with(_node: StartsWith, left: str, right: str) -> str:
    return f"STARTS_WITH({left}, {right})"


@format_operator_sql.register
def format_ends_with(_node: EndsWith, left: str, right: str) -> str:
    return f"ENDS_WITH({left}, {right})"


@format_operator_sql.register
def format_uppercase(_node: Uppercase, operand: str) -> str:
    return f"UPPER({operand})"


@format_operator_sql.register
def format_lowercase(_node: Lowercase, operand: str) -> str:
    return f"LOWER({operand})"
