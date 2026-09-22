from __future__ import annotations

import functools
import math
import unicodedata

from polars_bigquery.core.compiler.ir.base import (
    Column,
    Expr,
    ListLiteral,
    Literal,
    NullLiteral,
    Unsupported,
)
from polars_bigquery.core.compiler.ir.comparison import IsIn
from polars_bigquery.core.compiler.ir.numeric import FloatLiteral

_ALLOWED_FLEXIBLE_SPECIAL_CHARS = frozenset(
    {"&", "%", "=", "+", ":", "'", "<", ">", "#", "|"}
)
_UNSUPPORTED_COLUMN_CHARS = frozenset(
    {
        "!",
        '"',
        "$",
        "(",
        ")",
        "*",
        ",",
        ".",
        "/",
        ";",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "`",
        "{",
        "}",
        "~",
    }
)


def _is_allowed_unicode_category(ch: str) -> bool:
    """Check if character belongs to BigQuery's allowed Unicode categories.

    Per BigQuery flexible column name rules, the allowed Unicode categories are:
    - \\p{L} (Letters): category starts with 'L' (Lu, Ll, Lt, Lm, Lo)
    - \\p{N} (Numbers): category starts with 'N' (Nd, Nl, No)
    - \\p{M} (Marks): category starts with 'M' (Mn, Mc, Me) - accents, umlauts
    - \\p{Pc} (Connector Punctuation): underscores '_'
    - \\p{Pd} (Dash Punctuation): hyphens and dashes '-'
    """
    unicode_category = unicodedata.category(ch)
    return unicode_category.startswith(("L", "N", "M")) or unicode_category in (
        "Pc",
        "Pd",
    )


def _is_valid_column_character(ch: str) -> bool:
    """Check if character is permitted in a BigQuery standard or flexible column name."""
    if ch in _UNSUPPORTED_COLUMN_CHARS or ch in ("\n", "\r", "\0"):
        return False
    return (
        _is_allowed_unicode_category(ch)
        or ch in _ALLOWED_FLEXIBLE_SPECIAL_CHARS
        or ch.isspace()
    )


def column_to_sql_identifier(identifier: str) -> str:
    """Validate a BigQuery standard or flexible column name and wrap in backticks.

    Enforces BigQuery column naming character constraints (allowed Unicode
    character categories, whitespace, and permitted flexible symbols).

    Raises ValueError if the column name is empty or contains unsupported characters.
    """
    if not identifier:
        msg = f"Invalid BigQuery column name {identifier!r}: column name must not be empty."
        raise ValueError(msg)

    for ch in identifier:
        if not _is_valid_column_character(ch):
            msg = (
                f"Invalid BigQuery column name {identifier!r}: "
                f"contains unsupported character {ch!r}."
            )
            raise ValueError(msg)

    return f"`{identifier}`"


_column_to_sql_identifier = column_to_sql_identifier


@functools.singledispatch
def literal_ir_to_sql(node: Literal) -> str:
    """Convert a Literal IR node into a BigQuery SQL string."""
    msg = f"Unhandled literal IR node: {node!r}"
    raise TypeError(msg)


@literal_ir_to_sql.register
def format_null_literal(_node: NullLiteral) -> str:
    """Format a NullLiteral IR node as a BigQuery SQL NULL."""
    return "NULL"


@literal_ir_to_sql.register
def format_list_literal(node: ListLiteral) -> str:
    """Format a ListLiteral IR node as a parenthesized SQL tuple."""
    items_sql = ", ".join(literal_ir_to_sql(v) for v in node.values)
    return f"({items_sql})"


@functools.singledispatch
def format_operator_sql(node: Expr, *child_sqls: str) -> str | None:
    """Format SQL string for a non-monotone binary or unary operator node."""
    return None


@functools.singledispatch
def emit_node_sql(
    node: Expr, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    """Emit the BigQuery SQL string and exactness flag for a single IR node.

    Non-monotone / comparison / function operators require:
    1. All children to have valid non-None SQL representations.
    2. All children to be exact (is_exact=True), because relaxing a child inside
       NOT, =, !=, etc. flips subset/superset polarity and causes silent data loss.
    3. No child to be a bare NullLiteral (e.g. `col = NULL` evaluates to UNKNOWN in SQL).
    """
    if any(sql is None or not is_exact for sql, is_exact in child_results):
        return None, False
    if any(isinstance(child, NullLiteral) for child in node.children()):
        return None, False
    if isinstance(node, IsIn):
        if not isinstance(node.right, ListLiteral):
            return None, False
    elif any(isinstance(child, ListLiteral) for child in node.children()):
        return None, False

    child_sqls = [sql for sql, _ in child_results if sql is not None]
    formatted = format_operator_sql(node, *child_sqls)
    if formatted is not None:
        return formatted, True
    return None, False


@emit_node_sql.register
def emit_unsupported_sql(
    node: Unsupported, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    return None, False


@emit_node_sql.register
def emit_column_sql(
    node: Column, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    return column_to_sql_identifier(node.name), True


@emit_node_sql.register
def emit_literal_sql(
    node: Literal, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    if isinstance(node, ListLiteral) and (
        not node.values
        or any(
            isinstance(v, (NullLiteral, ListLiteral))
            or (isinstance(v, FloatLiteral) and math.isnan(v.value))
            for v in node.values
        )
    ):
        return None, False
    return literal_ir_to_sql(node), True
