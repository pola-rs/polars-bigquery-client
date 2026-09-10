from __future__ import annotations

import io
import json
from typing import Any

import polars as pl

_BINARY_OPS = {
    "Or": "OR",
    "And": "AND",
    "Eq": "=",
    "Gt": ">",
    "GtEq": ">=",
    "Lt": "<",
    "LtEq": "<=",
}


def _json_literal_to_sql(literal_json: dict[str, Any]) -> str | None:
    """Convert a literal from a polars expression JSON into SQL-like."""
    if "Dyn" in literal_json:
        return _json_literal_to_sql(literal_json["Dyn"])

    if "Scalar" in literal_json:
        return _json_literal_to_sql(literal_json["Scalar"])

    if "dtype" in literal_json and "value" in literal_json:
        return _json_literal_to_sql(literal_json["value"])

    polars_type, value = literal_json.popitem()

    # TODO: support more polars types.
    if polars_type == "DateTime":
        # TODO: check timezone, too
        # In BigQuery DATETIME is naive (no associated timezone) and TIMESTAMP is UTC.
        # https://stackoverflow.com/a/47724366/101923
        ticks = value[0]
        units = value[1]
        if units == "Microseconds":
            return f"TIMESTAMP_MICROS({ticks})"
        return None
    if polars_type == "Date":
        return f"DATE(TIMESTAMP_SECONDS({value} * 86400))"

    if polars_type in ("String", "Int", "Float"):
        return repr(value)

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
        input_ = _json_expr_to_row_restriction(inputs[0])
        if input_ is None:
            return None
        return f"({input_} IS NULL)"

    if boolean_function_name == "IsNotNull":
        if not inputs:
            return None
        input_ = _json_expr_to_row_restriction(inputs[0])
        if input_ is None:
            return None
        return f"({input_} IS NOT NULL)"

    if boolean_function_name == "Not":
        if not inputs:
            return None
        input_ = _json_expr_to_row_restriction(inputs[0])
        if input_ is None:
            return None
        return f"(NOT {input_})"

    string_function_name = function_details.get("StringExpr", None)
    if string_function_name == "StartsWith":
        if len(inputs) != 2:
            return None
        value = _json_expr_to_row_restriction(inputs[0])
        prefix = _json_expr_to_row_restriction(inputs[1])
        if value is None or prefix is None:
            return None
        return f"STARTS_WITH({value}, {prefix})"

    if string_function_name == "EndsWith":
        if len(inputs) != 2:
            return None
        value = _json_expr_to_row_restriction(inputs[0])
        suffix = _json_expr_to_row_restriction(inputs[1])
        if value is None or suffix is None:
            return None
        return f"ENDS_WITH({value}, {suffix})"

    return None


def _json_expr_to_row_restriction(expr_json: dict[str, Any]) -> str | None:
    """Create a row restriction to filter rows.

    Returns None if unknown operators are found and can't guarantee a superset of rows.
    """
    # TODO: Use iterative compilation to support deeper trees. Python 3.12+
    # has a pretty strict 1000 depth limit. See:
    # https://github.com/python/cpython/issues/112282
    if "BinaryExpr" in expr_json:
        binary_expr = expr_json["BinaryExpr"]
        left = _json_expr_to_row_restriction(binary_expr["left"])
        right = _json_expr_to_row_restriction(binary_expr["right"])

        polars_op = binary_expr.get("op", None)
        if polars_op is None:
            return None

        # TODO: lookup table instead of iterating through all possible types
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
        return _json_function_to_sql(function_json)

    if "Column" in expr_json:
        return f"`{expr_json['Column']}`"  # TODO: do we need to escape any characters?

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
    row_restriction = _json_expr_to_row_restriction(predicate_json)
    return row_restriction if row_restriction is not None else ""
