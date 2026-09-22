"""Functions for parsing Polars Expr JSON into an IR Expr tree."""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
from typing import Any

from polars_bigquery.core.compiler.ir import (
    BinaryExpr,
    Expr,
    Unsupported,
)
from polars_bigquery.core.compiler.parser import (
    base,
    boolean,
    comparison,
    list_,
    numeric,
    string,
    temporal,
)
from polars_bigquery.core.compiler.parser.base import (
    NULL_LITERAL_PARSERS,
    UNSUPPORTED_RECORD,
    parse_column_expr,
    unwrap_literal_json,
)
from polars_bigquery.core.compiler.parser.boolean import (
    BOOLEAN_LITERAL_PARSERS,
    LOGICAL_BINARY_OPS,
    parse_boolean_function,
)
from polars_bigquery.core.compiler.parser.comparison import (
    COMPARISON_BINARY_OPS,
)
from polars_bigquery.core.compiler.parser.list_ import (
    LIST_LITERAL_PARSERS,
)
from polars_bigquery.core.compiler.parser.numeric import (
    NUMERIC_LITERAL_PARSERS,
)
from polars_bigquery.core.compiler.parser.string import (
    STRING_LITERAL_PARSERS,
    parse_string_function,
)
from polars_bigquery.core.compiler.parser.temporal import (
    TEMPORAL_LITERAL_PARSERS,
)

# Keys represent Polars Rust AST `Operator` enum variant names (e.g., "And", "Eq")
# emitted in `{"BinaryExpr": {"op": "<key>"}}` when `pl.Expr.meta.serialize(format="json")`
# is called. Because `json.load` produces untyped `dict` and `str` objects (all keys have
# runtime type `str`), string-keyed lookup dictionaries are used here to construct IR
# dataclasses.
_BINARY_OPS: dict[str, type[BinaryExpr]] = {
    **LOGICAL_BINARY_OPS,
    **COMPARISON_BINARY_OPS,
}

# Keys represent Polars Rust AST `LiteralValue` / `AnyValue` enum variant names
# (e.g., "Int64", "StringOwned", "DateTime") emitted in `{"Literal": ...}` in serialized
# Polars JSON.
_LITERAL_PARSERS: dict[str, Callable[[Any], Expr]] = {
    **NULL_LITERAL_PARSERS,
    **BOOLEAN_LITERAL_PARSERS,
    **NUMERIC_LITERAL_PARSERS,
    **STRING_LITERAL_PARSERS,
    **TEMPORAL_LITERAL_PARSERS,
    **LIST_LITERAL_PARSERS,
}


def _json_literal_to_ir(literal_json: Any) -> Expr:
    """Convert a literal from a Polars expression JSON into an IR node.

    Depending on the Polars version and literal type, `literal_json` (the value
    under the `"Literal"` key in `pl.Expr.meta.serialize(format="json")`) may be:
    - A direct type-tagged scalar dictionary with a single entry:
      - `{"Int64": 42}` or `{"UInt32": 7}`
      - `{"Float64": 3.5}` or `{"Float64": "inf"}`
      - `{"String": "hello"}` or `{"StringOwned": "world"}`
      - `{"Boolean": True}`
      - `{"Null": None}`
      - `{"Date": 19723}` (days since UNIX epoch)
      - `{"DateTime": [1704110400000000, "Microseconds", "UTC"]}` or
        `{"Datetime": [1704110400000000, "Microseconds", None]}`
      - `{"List": [...]}` or `{"Series": [...]}` (IPC stream bytes)
    - A `Scalar` wrapper (Polars 1.x):
      `{"Scalar": {"dtype": "Int64", "value": {"Int64": 42}}}`
    - A `Dyn` wrapper for untyped literals:
      `{"Dyn": {"Int": 42}}` or `{"Dyn": {"Float": 3.5}}`
    """
    curr = unwrap_literal_json(literal_json)
    if not isinstance(curr, dict) or len(curr) != 1:
        return Unsupported()

    polars_type, value = next(iter(curr.items()))
    parser = _LITERAL_PARSERS.get(polars_type)
    if parser is not None:
        return parser(value)

    return Unsupported()


def _parse_binary_expr(binary_expr: Any) -> tuple[str, Any, tuple[Any, ...]]:
    """Extract IR constructor and child JSONs for a Polars `BinaryExpr` node."""
    if not isinstance(binary_expr, dict):
        return UNSUPPORTED_RECORD
    polars_op = binary_expr.get("op")
    binary_cls = _BINARY_OPS.get(polars_op) if isinstance(polars_op, str) else None
    left_json = binary_expr.get("left")
    right_json = binary_expr.get("right")
    if binary_cls is None or left_json is None or right_json is None:
        return UNSUPPORTED_RECORD
    return ("Binary", binary_cls, (left_json, right_json))


# Keys represent Polars Rust AST `FunctionExpr` enum variant names emitted under
# `{"Function": {"function": {"<key>": ...}}}` in serialized Polars JSON.
_FUNCTION_PARSERS: dict[
    str, Callable[[Any, list[Any]], tuple[str, Any, tuple[Any, ...]]]
] = {
    "Boolean": parse_boolean_function,
    "StringExpr": parse_string_function,
}


def _parse_function_expr(function_json: Any) -> tuple[str, Any, tuple[Any, ...]]:
    """Extract IR constructor and child JSONs for a Polars `Function` node."""
    if not isinstance(function_json, dict):
        return UNSUPPORTED_RECORD
    inputs = function_json.get("input", [])
    if not isinstance(inputs, list):
        return UNSUPPORTED_RECORD
    function_details = function_json.get("function")
    if isinstance(function_details, dict) and len(function_details) == 1:
        func_category, func_name = next(iter(function_details.items()))
        func_parser = _FUNCTION_PARSERS.get(func_category)
        if func_parser is not None:
            return func_parser(func_name, inputs)
    return UNSUPPORTED_RECORD


def _parse_literal_expr(literal_json: Any) -> tuple[str, Any, tuple[Any, ...]]:
    """Extract Literal leaf node for a Polars `Literal` node."""
    return ("Leaf", _json_literal_to_ir(literal_json), ())


# Keys represent Polars Rust AST `Expr` enum variant names (e.g., "BinaryExpr", "Function")
# emitted at each node in `pl.Expr.meta.serialize(format="json")`.
_EXPR_PARSERS: dict[str, Callable[[Any], tuple[str, Any, tuple[Any, ...]]]] = {
    "BinaryExpr": _parse_binary_expr,
    "Function": _parse_function_expr,
    "Column": parse_column_expr,
    "Literal": _parse_literal_expr,
}


def json_to_ir(expr_json: Any) -> Expr:
    """Compile Polars expression JSON into an immutable Expr IR tree.

    Performs a two-pass iterative compilation to avoid Python's recursion limit
    on deeply nested expression trees:
    1. Pass 1 (Top-down BFS traversal): Walks the JSON tree using a FIFO queue
       and records each node's kind, constructor (or leaf), and child indices.
    2. Pass 2 (Bottom-up IR construction): Iterates over the recorded nodes in
       reverse BFS order so every child `Expr` is instantiated before its parent.
    """
    # Pass 1: Top-down breadth-first discovery.
    # `raw_nodes` stores the raw JSON sub-expressions indexed by integer ID.
    # When node `curr_idx` is popped from `queue`, any child JSON sub-expressions
    # are appended to `raw_nodes` and `queue`. Because children are always appended
    # after their parent is popped, `parent_idx < child_idx` holds for every edge
    # in the expression tree, and `records[curr_idx]` stores the child indices.
    raw_nodes: list[Any] = [expr_json]
    records: list[tuple[str, Any, tuple[int, ...]]] = []
    queue: collections.deque[int] = collections.deque([0])

    while queue:
        curr_idx = queue.popleft()
        curr = raw_nodes[curr_idx]

        if not isinstance(curr, dict) or len(curr) != 1:
            records.append(UNSUPPORTED_RECORD)
            continue

        expr_type, payload = next(iter(curr.items()))
        parser = _EXPR_PARSERS.get(expr_type)
        if parser is None:
            records.append(UNSUPPORTED_RECORD)
            continue

        kind, cls_or_leaf, child_jsons = parser(payload)
        child_indices: Sequence[int] = []
        for child_json in child_jsons:
            child_idx = len(raw_nodes)
            raw_nodes.append(child_json)
            queue.append(child_idx)
            child_indices.append(child_idx)

        records.append((kind, cls_or_leaf, tuple(child_indices)))

    # Pass 2: Bottom-up IR tree construction.
    # Because Pass 1 guarantees `idx < c_idx` for all `c_idx` in `child_indices`,
    # iterating from `len(records) - 1` down to `0` ensures that `ir_nodes[c_idx]`
    # is already populated with a constructed frozen `Expr` dataclass before its
    # parent at `idx` is instantiated. `ir_nodes[0]` is the root `Expr`.
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


__all__ = [
    "base",
    "boolean",
    "comparison",
    "json_to_ir",
    "list_",
    "numeric",
    "string",
    "temporal",
]
