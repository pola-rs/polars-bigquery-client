from __future__ import annotations

import math
import unicodedata
from collections import deque
from collections.abc import Callable

from polars_bigquery.core.predicates.ir import (
    And,
    BinaryExpr,
    BoolLiteral,
    Column,
    DateLiteral,
    EndsWith,
    Eq,
    Expr,
    FloatLiteral,
    Gt,
    GtEq,
    IntLiteral,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
    IsNotNull,
    IsNull,
    Literal,
    Lt,
    LtEq,
    Not,
    NotEq,
    NullLiteral,
    Or,
    StartsWith,
    StringLiteral,
    TimestampLiteral,
    TimestampUnit,
    UnaryExpr,
    Unsupported,
)

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


def _escape_sql_string(val: str) -> str:
    escaped = (
        val.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\0", "\\x00")
    )
    return f"'{escaped}'"


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


def _column_to_sql_identifier(identifier: str) -> str:
    """Validate a BigQuery standard or flexible column name and wrap in backticks.

    Enforces BigQuery column naming constraints (1 to 300 characters, allowed
    Unicode character categories, whitespace, and permitted flexible symbols).

    Raises ValueError if the column name violates BigQuery column naming rules.
    """
    if not (1 <= len(identifier) <= 300):
        msg = (
            f"Invalid BigQuery column name length ({len(identifier)}): {identifier!r}. "
            "Column names must be between 1 and 300 characters."
        )
        raise ValueError(msg)

    for ch in identifier:
        if not _is_valid_column_character(ch):
            msg = (
                f"Invalid BigQuery column name {identifier!r}: "
                f"contains unsupported character {ch!r}."
            )
            raise ValueError(msg)

    return f"`{identifier}`"


def _literal_ir_to_sql(node: Literal) -> str:
    """Convert a Literal IR node into a BigQuery SQL string."""
    if isinstance(node, NullLiteral):
        return "NULL"
    if isinstance(node, BoolLiteral):
        return "TRUE" if node.value else "FALSE"
    if isinstance(node, IntLiteral):
        return str(node.value)
    if isinstance(node, FloatLiteral):
        if math.isnan(node.value):
            return "CAST('nan' AS FLOAT64)"
        if math.isinf(node.value):
            return (
                "CAST('-inf' AS FLOAT64)"
                if node.value < 0
                else "CAST('inf' AS FLOAT64)"
            )
        return repr(node.value)
    if isinstance(node, StringLiteral):
        return _escape_sql_string(node.value)
    if isinstance(node, DateLiteral):
        return f"DATE(TIMESTAMP_SECONDS({node.days} * 86400))"
    if isinstance(node, TimestampLiteral):
        if node.unit == TimestampUnit.MICROSECONDS:
            return f"TIMESTAMP_MICROS({node.ticks})"
        if node.unit == TimestampUnit.MILLISECONDS:
            return f"TIMESTAMP_MILLIS({node.ticks})"
    msg = f"Unhandled literal IR node: {node!r}"
    raise TypeError(msg)


_BINARY_SQL_FORMATTERS: dict[type[BinaryExpr], Callable[[str, str], str]] = {
    Or: lambda l, r: f"({l} OR {r})",
    Eq: lambda l, r: f"({l} = {r})",
    NotEq: lambda l, r: f"({l} != {r})",
    Gt: lambda l, r: f"({l} > {r})",
    GtEq: lambda l, r: f"({l} >= {r})",
    Lt: lambda l, r: f"({l} < {r})",
    LtEq: lambda l, r: f"({l} <= {r})",
    StartsWith: lambda l, r: f"STARTS_WITH({l}, {r})",
    EndsWith: lambda l, r: f"ENDS_WITH({l}, {r})",
}

_UNARY_SQL_FORMATTERS: dict[type[UnaryExpr], Callable[[str], str]] = {
    Not: lambda x: f"(NOT {x})",
    IsNull: lambda x: f"({x} IS NULL)",
    IsNotNull: lambda x: f"({x} IS NOT NULL)",
    IsNan: lambda x: f"IS_NAN({x})",
    IsNotNan: lambda x: f"(NOT IS_NAN({x}))",
    IsInfinite: lambda x: f"IS_INF({x})",
    IsFinite: lambda x: f"(NOT IS_INF({x}) AND NOT IS_NAN({x}))",
}


def _emit_node_sql(node: Expr, child_sqls: list[str | None]) -> str | None:
    """Emit the BigQuery SQL string for a single IR node given its compiled child SQL strings."""
    if isinstance(node, Unsupported):
        return None
    if isinstance(node, Column):
        return _column_to_sql_identifier(node.name)
    if isinstance(node, Literal):
        return _literal_ir_to_sql(node)

    if isinstance(node, And):
        left_sql, right_sql = child_sqls[0], child_sqls[1]
        if left_sql is None:
            return right_sql
        if right_sql is None:
            return left_sql
        return f"({left_sql} AND {right_sql})"

    # All remaining operators require all children to have valid SQL representations.
    if any(s is None for s in child_sqls):
        return None

    binary_formatter = _BINARY_SQL_FORMATTERS.get(type(node))
    if binary_formatter is not None:
        return binary_formatter(child_sqls[0], child_sqls[1])  # type: ignore[arg-type]

    unary_formatter = _UNARY_SQL_FORMATTERS.get(type(node))
    if unary_formatter is not None:
        return unary_formatter(child_sqls[0])  # type: ignore[arg-type]

    return None


def ir_to_sql(root: Expr) -> str | None:
    """Compile an Expr IR tree into a BigQuery SQL row restriction string.

    Performs a breadth-first walk using a queue to traverse the IR tree top-down,
    then evaluates SQL strings bottom-up in reverse BFS order to avoid stack size
    limitations. Returns None if the root expression cannot be safely converted.
    """
    nodes: list[Expr] = [root]
    children_map: list[tuple[int, ...]] = [()]
    queue: deque[int] = deque([0])

    while queue:
        curr_idx = queue.popleft()
        node = nodes[curr_idx]
        node_children = node.children()
        if node_children:
            child_indices: list[int] = []
            for child in node_children:
                c_idx = len(nodes)
                nodes.append(child)
                children_map.append(())
                queue.append(c_idx)
                child_indices.append(c_idx)
            children_map[curr_idx] = tuple(child_indices)

    sql_results: list[str | None] = [None] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        node = nodes[idx]
        child_sqls = [sql_results[c_idx] for c_idx in children_map[idx]]
        sql_results[idx] = _emit_node_sql(node, child_sqls)

    return sql_results[0]
