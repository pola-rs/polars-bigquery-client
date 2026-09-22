from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from polars_bigquery.core.compiler.ir.base import (
    BinaryExpr,
    Column,
    Expr,
    NullLiteral,
    UnaryExpr,
    Unsupported,
)

ParseRecord = tuple[str, Any, tuple[Any, ...]]
ParserResult = Expr | ParseRecord
ParserFunc = Callable[..., ParserResult]
_F = TypeVar("_F", bound=ParserFunc)

UNSUPPORTED_RECORD: ParseRecord = ("Leaf", Unsupported(), ())

# Registry mapping JSON Path expressions (e.g. "$.BinaryExpr.op.Eq",
# "$.Function.function.Boolean.Not", "$.Literal..Int64") to parser functions.
PARSERS: dict[str, ParserFunc] = {}


def parse_json_path(path: str) -> tuple[tuple[str, str], ...]:
    """Parse a dot-style JSON Path string into `(operator, segment)` steps.

    Supports direct child access (`.segment`) and recursive descent (`..segment`)
    rooted at `$` (e.g., `"$.BinaryExpr.op.Eq"`, `"$.Literal..Int64"`).
    """
    if not path.startswith("$."):
        msg = f"Invalid JSON path {path!r}: must start with '$.'"
        raise ValueError(msg)

    steps: list[tuple[str, str]] = []
    idx = 1
    n = len(path)
    while idx < n:
        if path.startswith("..", idx):
            op = ".."
            idx += 2
        elif path[idx] == ".":
            op = "."
            idx += 1
        else:
            msg = f"Invalid JSON path {path!r} at index {idx}"
            raise ValueError(msg)

        next_dot = path.find(".", idx)
        if next_dot == -1:
            segment = path[idx:]
            idx = n
        else:
            segment = path[idx:next_dot]
            idx = next_dot

        if not segment:
            msg = f"Invalid JSON path {path!r}: empty segment"
            raise ValueError(msg)
        steps.append((op, segment))

    return tuple(steps)


@dataclasses.dataclass
class _PathTrieNode:
    """Trie node for fast $O(1)$ per-step dispatching of registered JSON Paths."""

    children: dict[str, _PathTrieNode] = dataclasses.field(default_factory=dict)
    descendant_children: dict[str, _PathTrieNode] = dataclasses.field(
        default_factory=dict
    )
    parser: ParserFunc | None = None
    accepts_expr_json: bool = False


_DISPATCH_TRIE = _PathTrieNode()


def _insert_path_into_trie(root: _PathTrieNode, path: str, func: ParserFunc) -> None:
    """Insert a single JSON Path and its bound parser callable into `root`."""
    steps = parse_json_path(path)
    curr = root
    for op, segment in steps:
        target_map = curr.descendant_children if op == ".." else curr.children
        nxt = target_map.get(segment)
        if nxt is None:
            nxt = _PathTrieNode()
            target_map[segment] = nxt
        curr = nxt
    curr.parser = func
    curr.accepts_expr_json = len(inspect.signature(func).parameters) >= 2


def register_parser(*paths: str) -> Callable[[_F], _F]:
    """Decorator to register a parser function for one or more JSON Paths."""
    if not paths:
        msg = "register_parser requires at least one JSON Path string"
        raise ValueError(msg)

    def _decorator(func: _F) -> _F:
        for path in paths:
            PARSERS[path] = func
            _insert_path_into_trie(_DISPATCH_TRIE, path, func)
        return func

    return _decorator


def unwrap_literal_json(literal_json: Any) -> Any:
    """Unwrap Polars `Dyn`, `Scalar`, and `{"dtype": ..., "value": ...}` literal wrappers."""
    curr = literal_json
    while isinstance(curr, dict):
        if len(curr) == 1 and "Dyn" in curr:
            curr = curr["Dyn"]
        elif len(curr) == 1 and "Scalar" in curr:
            curr = curr["Scalar"]
        elif len(curr) == 2 and "dtype" in curr and "value" in curr:
            curr = curr["value"]
        else:
            break
    return curr


def _match_trie(
    root: _PathTrieNode, expr_json: Any
) -> tuple[_PathTrieNode, Any] | None:
    """Match `expr_json` against the compiled JSON Path trie."""
    if not isinstance(expr_json, dict) or len(expr_json) != 1:
        return None

    curr_node = root
    curr_val: Any = expr_json
    is_root = True

    while True:
        if (
            curr_node.parser is not None
            and not curr_node.children
            and not curr_node.descendant_children
        ):
            return curr_node, curr_val

        matched_next: tuple[_PathTrieNode, Any] | None = None

        if curr_node.children:
            if isinstance(curr_val, dict):
                if is_root or len(curr_val) == 1:
                    if len(curr_val) == 1:
                        k, v = next(iter(curr_val.items()))
                        child_node = curr_node.children.get(k)
                        if child_node is not None:
                            matched_next = (child_node, v)
                else:
                    for k, child_node in curr_node.children.items():
                        if child_node.parser is None and k in curr_val:
                            matched_next = (child_node, curr_val[k])
                            break
            elif isinstance(curr_val, str):
                child_node = curr_node.children.get(curr_val)
                if child_node is not None:
                    matched_next = (child_node, curr_val)

        if matched_next is None and curr_node.descendant_children:
            unwrapped = unwrap_literal_json(curr_val)
            if isinstance(unwrapped, dict) and len(unwrapped) == 1:
                k, v = next(iter(unwrapped.items()))
                child_node = curr_node.descendant_children.get(k)
                if child_node is not None:
                    matched_next = (child_node, v)

        if matched_next is None:
            if curr_node.parser is not None:
                return curr_node, curr_val
            return None

        curr_node, curr_val = matched_next
        is_root = False


def dispatch_parser(expr_json: Any) -> ParseRecord:
    """Dispatch a Polars expression JSON node to its matching registered parser.

    Returns `UNSUPPORTED_RECORD` (`("Leaf", Unsupported(), ())`) if no registered
    JSON Path matches `expr_json`.
    """
    matched = _match_trie(_DISPATCH_TRIE, expr_json)
    if matched is None:
        return UNSUPPORTED_RECORD

    node, matched_val = matched
    parser = node.parser
    if parser is None:
        return UNSUPPORTED_RECORD

    result = (
        parser(matched_val, expr_json)
        if node.accepts_expr_json
        else parser(matched_val)
    )
    if isinstance(result, Expr):
        return ("Leaf", result, ())
    return result


def extract_binary_operands(expr_json: Any) -> tuple[Any, Any] | None:
    """Extract `(left, right)` operand JSONs from a Polars `BinaryExpr` JSON node."""
    if isinstance(expr_json, dict):
        binary_body = expr_json.get("BinaryExpr", expr_json)
        if isinstance(binary_body, dict):
            left_json = binary_body.get("left")
            right_json = binary_body.get("right")
            if left_json is not None and right_json is not None:
                return left_json, right_json
    return None


def parse_binary_op(expr_json: Any, binary_cls: type[BinaryExpr]) -> ParseRecord:
    """Build a Binary `ParseRecord` for `binary_cls` from a `BinaryExpr` JSON node."""
    operands = extract_binary_operands(expr_json)
    if operands is None:
        return UNSUPPORTED_RECORD
    return ("Binary", binary_cls, operands)


def extract_function_inputs(expr_json: Any) -> list[Any] | None:
    """Extract the `input` list from a Polars `Function` JSON node or raw input list."""
    if isinstance(expr_json, list):
        return expr_json
    if isinstance(expr_json, dict):
        func_body = expr_json.get("Function", expr_json)
        if isinstance(func_body, dict):
            inputs = func_body.get("input", [])
            if isinstance(inputs, list):
                return inputs
    return None


def parse_unary_function(
    expr_json: Any,
    unary_cls: type[UnaryExpr],
    *,
    exact_inputs: bool = False,
) -> ParseRecord:
    """Build a Unary `ParseRecord` for `unary_cls` from a `Function` JSON node."""
    inputs = extract_function_inputs(expr_json)
    if inputs is None:
        return UNSUPPORTED_RECORD
    if (exact_inputs and len(inputs) != 1) or len(inputs) < 1:
        return UNSUPPORTED_RECORD
    return ("Unary", unary_cls, (inputs[0],))


def parse_binary_function(
    expr_json: Any,
    binary_cls: Any,
) -> ParseRecord:
    """Build a Binary `ParseRecord` for `binary_cls` from a 2-input `Function` JSON node."""
    inputs = extract_function_inputs(expr_json)
    if inputs is None or len(inputs) != 2:
        return UNSUPPORTED_RECORD
    return ("Binary", binary_cls, (inputs[0], inputs[1]))


@register_parser("$.Literal..Null")
def parse_null_literal(_value: Any) -> NullLiteral:
    """Parse a Null value from Polars JSON into a NullLiteral."""
    return NullLiteral()


@register_parser("$.Column")
def parse_column_expr(column_json: Any) -> ParseRecord:
    """Extract Column leaf node for a Polars `Column` node."""
    return ("Leaf", Column(name=str(column_json)), ())
