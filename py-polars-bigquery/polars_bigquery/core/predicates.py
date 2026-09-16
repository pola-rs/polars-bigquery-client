from __future__ import annotations

import io
import json
import math
import unicodedata
from typing import Any

import polars as pl

_BINARY_OPS = {
    "Or": "OR",
    "And": "AND",
    "Eq": "=",
    "NotEq": "!=",
    "Gt": ">",
    "GtEq": ">=",
    "Lt": "<",
    "LtEq": "<=",
}

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
    return (
        unicode_category.startswith(("L", "N", "M"))
        or unicode_category in ("Pc", "Pd")
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


_INT_TYPES = frozenset(
    {
        "Int",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    }
)
_FLOAT_TYPES = frozenset({"Float", "Float32", "Float64"})
_STRING_TYPES = frozenset({"String", "StringOwned"})


def _json_literal_to_sql(literal_json: dict[str, Any]) -> str | None:
    """Convert a literal from a polars expression JSON into SQL-like."""
    if not isinstance(literal_json, dict) or not literal_json:
        return None

    if "Dyn" in literal_json:
        return _json_literal_to_sql(literal_json["Dyn"])

    if "Scalar" in literal_json:
        return _json_literal_to_sql(literal_json["Scalar"])

    if "dtype" in literal_json and "value" in literal_json:
        return _json_literal_to_sql(literal_json["value"])

    polars_type, value = next(iter(literal_json.items()))

    if polars_type == "DateTime":
        # In BigQuery DATETIME is naive (no associated timezone) and TIMESTAMP is UTC.
        try:
            ticks = int(value[0])
            units = value[1]
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        if units == "Microseconds":
            return f"TIMESTAMP_MICROS({ticks})"
        if units == "Milliseconds":
            return f"TIMESTAMP_MILLIS({ticks})"
        if units == "Nanoseconds" and ticks % 1000 == 0:
            return f"TIMESTAMP_MICROS({ticks // 1000})"
        return None
    if polars_type == "Date":
        try:
            days = int(value)
        except (TypeError, ValueError):
            return None
        return f"DATE(TIMESTAMP_SECONDS({days} * 86400))"

    if polars_type == "Boolean":
        return "TRUE" if value else "FALSE"

    if polars_type == "Null":
        return "NULL"

    if polars_type in _STRING_TYPES:
        return _escape_sql_string(str(value))

    if polars_type in _INT_TYPES:
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return None

    if polars_type in _FLOAT_TYPES:
        if value is None:
            return None
        if isinstance(value, str):
            val_lower = value.strip().lower()
            if val_lower in ("nan", "+nan", "-nan"):
                return "CAST('nan' AS FLOAT64)"
            if val_lower in ("inf", "+inf", "infinity", "+infinity"):
                return "CAST('inf' AS FLOAT64)"
            if val_lower in ("-inf", "-infinity"):
                return "CAST('-inf' AS FLOAT64)"
        try:
            float_val = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(float_val):
            return "CAST('nan' AS FLOAT64)"
        if math.isinf(float_val):
            return "CAST('-inf' AS FLOAT64)" if float_val < 0 else "CAST('inf' AS FLOAT64)"
        return repr(float_val)

    return None


def _json_function_to_sql(function_json: dict[str, Any]) -> str | None:
    """Converts a polars function call into the equivalent BigQuery syntax."""
    function_details = function_json.get("function")
    if not isinstance(function_details, dict):
        return None

    inputs = function_json.get("input", [])

    # So far, only boolean output functions are supported.
    boolean_function_name = function_details.get("Boolean", None)
    if boolean_function_name == "IsNull":
        if not inputs:
            return None
        try:
            input_ = _json_expr_to_row_restriction(inputs[0])
        except RecursionError:
            return None
        if input_ is None:
            return None
        return f"({input_} IS NULL)"

    if boolean_function_name == "IsNotNull":
        if not inputs:
            return None
        try:
            input_ = _json_expr_to_row_restriction(inputs[0])
        except RecursionError:
            return None
        if input_ is None:
            return None
        return f"({input_} IS NOT NULL)"

    if boolean_function_name == "IsNan":
        if not inputs:
            return None
        try:
            input_ = _json_expr_to_row_restriction(inputs[0])
        except RecursionError:
            return None
        if input_ is None:
            return None
        return f"IS_NAN({input_})"

    if boolean_function_name == "IsNotNan":
        if not inputs:
            return None
        try:
            input_ = _json_expr_to_row_restriction(inputs[0])
        except RecursionError:
            return None
        if input_ is None:
            return None
        return f"(NOT IS_NAN({input_}))"

    if boolean_function_name == "IsInfinite":
        if not inputs:
            return None
        try:
            input_ = _json_expr_to_row_restriction(inputs[0])
        except RecursionError:
            return None
        if input_ is None:
            return None
        return f"IS_INF({input_})"

    if boolean_function_name == "IsFinite":
        if not inputs:
            return None
        try:
            input_ = _json_expr_to_row_restriction(inputs[0])
        except RecursionError:
            return None
        if input_ is None:
            return None
        return f"(NOT IS_INF({input_}) AND NOT IS_NAN({input_}))"

    if boolean_function_name == "Not":
        if not inputs:
            return None
        try:
            input_ = _json_expr_to_row_restriction(inputs[0])
        except RecursionError:
            return None
        if input_ is None:
            return None
        return f"(NOT {input_})"

    string_function_name = function_details.get("StringExpr", None)
    if string_function_name == "StartsWith":
        if len(inputs) != 2:
            return None
        try:
            value = _json_expr_to_row_restriction(inputs[0])
            prefix = _json_expr_to_row_restriction(inputs[1])
        except RecursionError:
            return None
        if value is None or prefix is None:
            return None
        return f"STARTS_WITH({value}, {prefix})"

    if string_function_name == "EndsWith":
        if len(inputs) != 2:
            return None
        try:
            value = _json_expr_to_row_restriction(inputs[0])
            suffix = _json_expr_to_row_restriction(inputs[1])
        except RecursionError:
            return None
        if value is None or suffix is None:
            return None
        return f"ENDS_WITH({value}, {suffix})"

    return None


def _json_expr_to_row_restriction(expr_json: dict[str, Any]) -> str | None:
    """Create a row restriction to filter rows.

    Returns None if unknown operators are found and can't guarantee a superset of rows.
    """
    if not isinstance(expr_json, dict):
        return None

    if "BinaryExpr" in expr_json:
        binary_expr = expr_json["BinaryExpr"]
        if not isinstance(binary_expr, dict):
            return None
        left_json = binary_expr.get("left")
        right_json = binary_expr.get("right")
        if left_json is None or right_json is None:
            return None

        try:
            left = _json_expr_to_row_restriction(left_json)
        except RecursionError:
            left = None

        try:
            right = _json_expr_to_row_restriction(right_json)
        except RecursionError:
            right = None

        polars_op = binary_expr.get("op", None)
        if polars_op is None:
            return None

        if polars_op == "And":
            # With 'And', filtering by just one of the two children will still
            # give a superset of the filtered rows. The rest of the filters can
            # be applied by polars instead of BigQuery.
            if left is None:
                return right
            if right is None:
                return left

            return f"({left} AND {right})"

        # The rest of these operators need both left and right to be converted
        # correctly for correctness.
        if left is None or right is None:
            return None

        sql_op = _BINARY_OPS.get(polars_op)
        if sql_op is None:
            return None
        return f"({left} {sql_op} {right})"

    if "Function" in expr_json:
        function_json = expr_json["Function"]
        if not isinstance(function_json, dict):
            return None
        return _json_function_to_sql(function_json)

    if "Column" in expr_json:
        return _column_to_sql_identifier(str(expr_json["Column"]))

    if "Literal" in expr_json:
        literal = expr_json["Literal"]
        return _json_literal_to_sql(literal)

    # Got some op that we don't know how to handle.
    return None


def predicate_to_row_restriction(predicate: pl.Expr) -> str:
    predicate_json_file = io.BytesIO()
    predicate.meta.serialize(predicate_json_file, format="json")
    predicate_json_file.seek(0)
    predicate_json = json.load(predicate_json_file)
    try:
        row_restriction = _json_expr_to_row_restriction(predicate_json)
    except RecursionError:
        row_restriction = None
    return row_restriction if row_restriction is not None else ""
