from __future__ import annotations

import collections
import functools
import math
import unicodedata
from typing import Any

from polars_bigquery.core.predicates.ir import (
    And,
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
    Unsupported,
    _json_literal_to_ir,
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


@functools.singledispatch
def _literal_ir_to_sql(node: Literal) -> str:
    """Convert a Literal IR node into a BigQuery SQL string."""
    msg = f"Unhandled literal IR node: {node!r}"
    raise TypeError(msg)


def _json_literal_to_sql(literal_json: dict[str, Any]) -> str | None:
    """Convert a literal from a Polars expression JSON into a BigQuery SQL string."""
    ir_node = _json_literal_to_ir(literal_json)
    if isinstance(ir_node, Literal):
        return _literal_ir_to_sql(ir_node)
    return None


@_literal_ir_to_sql.register
def _(node: NullLiteral) -> str:
    return "NULL"


@_literal_ir_to_sql.register
def _(node: BoolLiteral) -> str:
    return "TRUE" if node.value else "FALSE"


@_literal_ir_to_sql.register
def _(node: IntLiteral) -> str:
    return str(node.value)


@_literal_ir_to_sql.register
def _(node: FloatLiteral) -> str:
    if math.isnan(node.value):
        return "CAST('nan' AS FLOAT64)"
    if math.isinf(node.value):
        return "CAST('-inf' AS FLOAT64)" if node.value < 0 else "CAST('inf' AS FLOAT64)"
    return repr(node.value)


@_literal_ir_to_sql.register
def _(node: StringLiteral) -> str:
    return _escape_sql_string(node.value)


@_literal_ir_to_sql.register
def _(node: DateLiteral) -> str:
    return f"DATE(TIMESTAMP_SECONDS({node.days} * 86400))"


@_literal_ir_to_sql.register
def _(node: TimestampLiteral) -> str:
    if node.unit == TimestampUnit.MICROSECONDS:
        ts_sql = f"TIMESTAMP_MICROS({node.ticks})"
    elif node.unit == TimestampUnit.MILLISECONDS:
        ts_sql = f"TIMESTAMP_MILLIS({node.ticks})"
    else:
        msg = f"Unhandled timestamp unit: {node.unit!r}"
        raise TypeError(msg)
    return f"DATETIME({ts_sql})" if node.tz is None else ts_sql


@functools.singledispatch
def _format_operator_sql(node: Expr, *child_sqls: str) -> str | None:
    """Format SQL string for a non-monotone binary or unary operator node."""
    return None


@_format_operator_sql.register
def _(node: Eq, left: str, right: str) -> str:
    return f"({left} = {right})"


@_format_operator_sql.register
def _(node: NotEq, left: str, right: str) -> str:
    return f"({left} != {right})"


@_format_operator_sql.register
def _(node: Gt, left: str, right: str) -> str:
    return f"({left} > {right})"


@_format_operator_sql.register
def _(node: GtEq, left: str, right: str) -> str:
    return f"({left} >= {right})"


@_format_operator_sql.register
def _(node: Lt, left: str, right: str) -> str:
    return f"({left} < {right})"


@_format_operator_sql.register
def _(node: LtEq, left: str, right: str) -> str:
    return f"({left} <= {right})"


@_format_operator_sql.register
def _(node: StartsWith, left: str, right: str) -> str:
    return f"STARTS_WITH({left}, {right})"


@_format_operator_sql.register
def _(node: EndsWith, left: str, right: str) -> str:
    return f"ENDS_WITH({left}, {right})"


@_format_operator_sql.register
def _(node: Not, operand: str) -> str:
    return f"(NOT {operand})"


@_format_operator_sql.register
def _(node: IsNull, operand: str) -> str:
    return f"({operand} IS NULL)"


@_format_operator_sql.register
def _(node: IsNotNull, operand: str) -> str:
    return f"({operand} IS NOT NULL)"


@_format_operator_sql.register
def _(node: IsNan, operand: str) -> str:
    return f"IS_NAN({operand})"


@_format_operator_sql.register
def _(node: IsNotNan, operand: str) -> str:
    return f"(NOT IS_NAN({operand}))"


@_format_operator_sql.register
def _(node: IsInfinite, operand: str) -> str:
    return f"IS_INF({operand})"


@_format_operator_sql.register
def _(node: IsFinite, operand: str) -> str:
    return f"(NOT IS_INF({operand}) AND NOT IS_NAN({operand}))"


@functools.singledispatch
def _emit_node_sql(
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

    child_sqls = [sql for sql, _ in child_results if sql is not None]
    formatted = _format_operator_sql(node, *child_sqls)
    if formatted is not None:
        return formatted, True
    return None, False


@_emit_node_sql.register
def _(
    node: Unsupported, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    return None, False


@_emit_node_sql.register
def _(
    node: Column, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    return _column_to_sql_identifier(node.name), True


@_emit_node_sql.register
def _(
    node: Literal, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    return _literal_ir_to_sql(node), True


@_emit_node_sql.register
def _(
    node: And, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    (left_sql, left_exact), (right_sql, right_exact) = (
        child_results[0],
        child_results[1],
    )
    if left_sql is None and right_sql is None:
        return None, False
    if left_sql is None:
        return right_sql, False
    if right_sql is None:
        return left_sql, False
    return f"({left_sql} AND {right_sql})", (left_exact and right_exact)


@_emit_node_sql.register
def _(
    node: Or, child_results: list[tuple[str | None, bool]]
) -> tuple[str | None, bool]:
    (left_sql, left_exact), (right_sql, right_exact) = (
        child_results[0],
        child_results[1],
    )
    if left_sql is None or right_sql is None:
        return None, False
    return f"({left_sql} OR {right_sql})", (left_exact and right_exact)


def _compile_ir_to_sql(root: Expr) -> tuple[str | None, bool]:
    """Compile an Expr IR tree into a BigQuery SQL string and an exactness boolean.

    Performs a breadth-first walk using a queue to traverse the IR tree top-down,
    then evaluates SQL strings bottom-up in reverse BFS order to avoid stack size
    limitations.
    """
    nodes: list[Expr] = [root]
    children_map: list[tuple[int, ...]] = [()]
    queue: collections.deque[int] = collections.deque([0])

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

    results: list[tuple[str | None, bool]] = [(None, False)] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        node = nodes[idx]
        child_results = [results[c_idx] for c_idx in children_map[idx]]
        results[idx] = _emit_node_sql(node, child_results)

    return results[0]


def ir_to_sql(root: Expr) -> str | None:
    """Compile an Expr IR tree into a BigQuery SQL row restriction string.

    Returns None if the root expression cannot be safely converted.
    """
    sql, _is_exact = _compile_ir_to_sql(root)
    return sql
