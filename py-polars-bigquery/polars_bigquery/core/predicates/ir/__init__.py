from __future__ import annotations

from collections import deque
from typing import Any

from polars_bigquery.core.predicates.ir.base import (
    BinaryExpr,
    Column,
    Expr,
    Literal,
    NullLiteral,
    UnaryExpr,
    Unsupported,
)
from polars_bigquery.core.predicates.ir.boolean import (
    BOOLEAN_UNARY_FUNCTIONS,
    LOGICAL_BINARY_OPS,
    And,
    BoolLiteral,
    IsNotNull,
    IsNull,
    Not,
    Or,
)
from polars_bigquery.core.predicates.ir.comparison import (
    COMPARISON_BINARY_OPS,
    Eq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    NotEq,
)
from polars_bigquery.core.predicates.ir.numeric import (
    FLOAT_TYPES,
    INT_TYPES,
    NUMERIC_UNARY_FUNCTIONS,
    FloatLiteral,
    IntLiteral,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
    parse_float_literal,
    parse_int_literal,
)
from polars_bigquery.core.predicates.ir.string import (
    STRING_BINARY_FUNCTIONS,
    STRING_TYPES,
    EndsWith,
    StartsWith,
    StringLiteral,
    parse_string_literal,
)
from polars_bigquery.core.predicates.ir.temporal import (
    DateLiteral,
    TimestampLiteral,
    TimestampUnit,
    parse_date_literal,
    parse_datetime_literal,
)

__all__ = [
    "And",
    "BinaryExpr",
    "BoolLiteral",
    "Column",
    "DateLiteral",
    "EndsWith",
    "Eq",
    "Expr",
    "FloatLiteral",
    "Gt",
    "GtEq",
    "IntLiteral",
    "IsFinite",
    "IsInfinite",
    "IsNan",
    "IsNotNan",
    "IsNotNull",
    "IsNull",
    "Literal",
    "Lt",
    "LtEq",
    "Not",
    "NotEq",
    "NullLiteral",
    "Or",
    "StartsWith",
    "StringLiteral",
    "TimestampLiteral",
    "TimestampUnit",
    "UnaryExpr",
    "Unsupported",
    "_json_literal_to_ir",
    "json_to_ir",
]

# Keys represent Polars Rust AST `Operator` enum variant names (e.g., "And", "Eq")
# emitted in `{"BinaryExpr": {"op": "<key>"}}` when `pl.Expr.meta.serialize(format="json")`
# is called. Because `json.load` produces untyped `dict` and `str` objects (all keys have
# runtime type `str`), string-keyed lookup dictionaries are used here to construct IR
# dataclasses.
_BINARY_OPS: dict[str, type[BinaryExpr]] = {
    **LOGICAL_BINARY_OPS,
    **COMPARISON_BINARY_OPS,
}

# Keys represent Polars Rust AST `BooleanFunction` enum variant names (e.g., "Not", "IsNull")
# emitted in `{"Function": {"function": {"Boolean": "<key>"}}}` in serialized Polars JSON.
_BOOLEAN_FUNCTIONS: dict[str, type[UnaryExpr]] = {
    **BOOLEAN_UNARY_FUNCTIONS,
    **NUMERIC_UNARY_FUNCTIONS,
}


def _json_literal_to_ir(literal_json: Any) -> Expr:
    """Convert a literal from a Polars expression JSON into an IR node."""
    curr = literal_json
    while isinstance(curr, dict):
        if "Dyn" in curr:
            curr = curr["Dyn"]
        elif "Scalar" in curr:
            curr = curr["Scalar"]
        elif "dtype" in curr and "value" in curr:
            curr = curr["value"]
        else:
            break

    if not isinstance(curr, dict) or not curr:
        return Unsupported()

    polars_type, value = next(iter(curr.items()))

    if polars_type in ("Datetime", "DateTime"):
        return parse_datetime_literal(value)
    if polars_type == "Date":
        return parse_date_literal(value)
    if polars_type == "Boolean":
        return BoolLiteral(value=bool(value))
    if polars_type == "Null":
        return NullLiteral()
    if polars_type in STRING_TYPES:
        return parse_string_literal(value)
    if polars_type in INT_TYPES:
        return parse_int_literal(value)
    if polars_type in FLOAT_TYPES:
        return parse_float_literal(value)

    return Unsupported()


def json_to_ir(expr_json: Any) -> Expr:
    """Compile Polars expression JSON into an immutable Expr IR tree.

    Performs a breadth-first walk using a queue to discover nodes top-down,
    then constructs frozen dataclass IR nodes bottom-up in reverse BFS order
    to avoid stack size limitations.
    """
    raw_nodes: list[Any] = [expr_json]
    records: list[tuple[str, Any, tuple[int, ...]]] = []
    queue: deque[int] = deque([0])

    while queue:
        curr_idx = queue.popleft()
        curr = raw_nodes[curr_idx]

        if not isinstance(curr, dict):
            records.append(("Leaf", Unsupported(), ()))
            continue

        if "BinaryExpr" in curr:
            binary_expr = curr["BinaryExpr"]
            if not isinstance(binary_expr, dict):
                records.append(("Leaf", Unsupported(), ()))
                continue
            polars_op = binary_expr.get("op")
            binary_cls = (
                _BINARY_OPS.get(polars_op) if isinstance(polars_op, str) else None
            )
            left_json = binary_expr.get("left")
            right_json = binary_expr.get("right")
            if binary_cls is None or left_json is None or right_json is None:
                records.append(("Leaf", Unsupported(), ()))
                continue

            left_idx = len(raw_nodes)
            raw_nodes.append(left_json)
            queue.append(left_idx)

            right_idx = len(raw_nodes)
            raw_nodes.append(right_json)
            queue.append(right_idx)

            records.append(("Binary", binary_cls, (left_idx, right_idx)))
            continue

        if "Function" in curr:
            function_json = curr["Function"]
            if not isinstance(function_json, dict):
                records.append(("Leaf", Unsupported(), ()))
                continue
            inputs = function_json.get("input", [])
            if not isinstance(inputs, list):
                records.append(("Leaf", Unsupported(), ()))
                continue

            function_details = function_json.get("function")
            if isinstance(function_details, dict):
                boolean_name = function_details.get("Boolean")
                if (
                    isinstance(boolean_name, str)
                    and boolean_name in _BOOLEAN_FUNCTIONS
                    and len(inputs) >= 1
                ):
                    child_idx = len(raw_nodes)
                    raw_nodes.append(inputs[0])
                    queue.append(child_idx)
                    records.append(
                        (
                            "Unary",
                            _BOOLEAN_FUNCTIONS[boolean_name],
                            (child_idx,),
                        )
                    )
                    continue

                string_name = function_details.get("StringExpr")
                if (
                    isinstance(string_name, str)
                    and string_name in STRING_BINARY_FUNCTIONS
                    and len(inputs) == 2
                ):
                    left_idx = len(raw_nodes)
                    raw_nodes.append(inputs[0])
                    queue.append(left_idx)
                    right_idx = len(raw_nodes)
                    raw_nodes.append(inputs[1])
                    queue.append(right_idx)
                    records.append(
                        (
                            "Binary",
                            STRING_BINARY_FUNCTIONS[string_name],
                            (left_idx, right_idx),
                        )
                    )
                    continue

            records.append(("Leaf", Unsupported(), ()))
            continue

        if "Column" in curr:
            records.append(("Leaf", Column(name=str(curr["Column"])), ()))
            continue

        if "Literal" in curr:
            records.append(("Leaf", _json_literal_to_ir(curr["Literal"]), ()))
            continue

        records.append(("Leaf", Unsupported(), ()))

    ir_nodes: list[Expr] = [Unsupported()] * len(records)
    for idx in range(len(records) - 1, -1, -1):
        kind, cls_or_leaf, child_indices = records[idx]
        if kind == "Leaf":
            ir_nodes[idx] = cls_or_leaf
        elif kind == "Unary":
            ir_nodes[idx] = cls_or_leaf(expr=ir_nodes[child_indices[0]])
        elif kind == "Binary":
            ir_nodes[idx] = cls_or_leaf(
                left=ir_nodes[child_indices[0]],
                right=ir_nodes[child_indices[1]],
            )
        else:
            ir_nodes[idx] = Unsupported(
                operands=tuple(ir_nodes[c_idx] for c_idx in child_indices)
            )

    return ir_nodes[0]
