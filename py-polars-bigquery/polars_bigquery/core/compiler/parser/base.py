from __future__ import annotations

import dataclasses
import inspect
import types
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, TypeVar

from polars_bigquery.core.compiler.ir.base import (
    BinaryExpr,
    Column,
    Expr,
    NullLiteral,
    UnaryExpr,
    Unsupported,
)

RecordKind = Literal["Leaf", "Unary", "Binary"]


class ParseRecord(NamedTuple):
    """Strongly-typed AST parse record compatible with 3-tuple unpacking."""

    kind: RecordKind
    node: Any
    children: tuple[Any, ...]


ParserResult = Expr | ParseRecord
ParserFunc = Callable[..., ParserResult]
_F = TypeVar("_F", bound=ParserFunc)

UNSUPPORTED_RECORD: ParseRecord = ParseRecord("Leaf", Unsupported(), ())
MAX_TRIE_DEPTH = 16

# Registry mapping JSON Path expressions (e.g. "$.BinaryExpr.op.Eq",
# "$.Function.function.Boolean.Not", "$.Literal..Int64") to parser functions.
_PARSERS_REGISTRY: dict[str, ParserFunc] = {}
PARSERS: types.MappingProxyType[str, ParserFunc] = types.MappingProxyType(
    _PARSERS_REGISTRY
)
_BUILTIN_PARSERS_LOADED = False


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
    """Trie node for fast per-step dispatching of registered JSON Paths."""

    children: dict[str, _PathTrieNode] = dataclasses.field(default_factory=dict)
    descendant_children: dict[str, _PathTrieNode] = dataclasses.field(
        default_factory=dict
    )
    parser: ParserFunc | None = None
    accepts_expr_json: bool = False


_DISPATCH_TRIE = _PathTrieNode()


def _accepts_second_positional_arg(func: ParserFunc) -> bool:
    """Return True if `func` can accept at least 2 positional arguments."""
    positional_count = 0
    for param in inspect.signature(func).parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
    return positional_count >= 2


def _insert_steps_into_trie(
    root: _PathTrieNode,
    path: str,
    steps: tuple[tuple[str, str], ...],
    func: ParserFunc,
) -> None:
    """Insert a pre-parsed JSON Path and its bound parser callable into `root`."""
    curr = root
    for op, segment in steps:
        target_map = curr.descendant_children if op == ".." else curr.children
        nxt = target_map.get(segment)
        if nxt is None:
            nxt = _PathTrieNode()
            target_map[segment] = nxt
        curr = nxt
    if curr.parser is not None and curr.parser is not func:
        msg = (
            f"Duplicate parser registration for JSON path {path!r}: "
            f"{curr.parser!r} vs {func!r}"
        )
        raise ValueError(msg)
    curr.parser = func
    curr.accepts_expr_json = _accepts_second_positional_arg(func)


def register_parser(*paths: str) -> Callable[[_F], _F]:
    """Decorator to register a parser function for one or more JSON Paths."""
    if not paths:
        msg = "register_parser requires at least one JSON Path string"
        raise ValueError(msg)

    parsed_paths = [(path, parse_json_path(path)) for path in paths]

    def _decorator(func: _F) -> _F:
        for path, _ in parsed_paths:
            existing = _PARSERS_REGISTRY.get(path)
            if existing is not None and existing is not func:
                msg = (
                    f"Duplicate parser registration for JSON path {path!r}: "
                    f"{existing!r} vs {func!r}"
                )
                raise ValueError(msg)
        for path, steps in parsed_paths:
            _insert_steps_into_trie(_DISPATCH_TRIE, path, steps, func)
            _PARSERS_REGISTRY[path] = func
        return func

    return _decorator


def _unwrap_literal_with_depth(literal_json: Any, depth: int) -> tuple[Any, int]:
    """Unwrap Polars `Dyn`, `Scalar`, and `{"dtype": ..., "value": ...}` wrappers within `MAX_TRIE_DEPTH`."""
    curr = literal_json
    curr_depth = depth
    while isinstance(curr, dict) and curr_depth <= MAX_TRIE_DEPTH:
        if len(curr) == 1 and "Dyn" in curr:
            curr = curr["Dyn"]
            curr_depth += 1
        elif len(curr) == 1 and "Scalar" in curr:
            curr = curr["Scalar"]
            curr_depth += 1
        elif len(curr) == 2 and "dtype" in curr and "value" in curr:
            curr = curr["value"]
            curr_depth += 1
        else:
            break
    return curr, curr_depth


def unwrap_literal_json(literal_json: Any) -> Any:
    """Unwrap Polars `Dyn`, `Scalar`, and `{"dtype": ..., "value": ...}` literal wrappers."""
    unwrapped, unwrapped_depth = _unwrap_literal_with_depth(literal_json, depth=0)
    if unwrapped_depth > MAX_TRIE_DEPTH:
        return None
    return unwrapped


def _match_descendant(
    descendant_map: dict[str, _PathTrieNode],
    curr_val: Any,
    *,
    depth: int,
) -> tuple[_PathTrieNode, Any] | None:
    """Perform bounded recursive descent (`..`) across wrappers, dicts, and unit strings."""
    if depth > MAX_TRIE_DEPTH:
        return None

    unwrapped, unwrapped_depth = _unwrap_literal_with_depth(curr_val, depth)
    if unwrapped_depth > MAX_TRIE_DEPTH:
        return None

    if isinstance(unwrapped, str):
        child_node = descendant_map.get(unwrapped)
        if child_node is not None:
            matched = _match_node(
                child_node, unwrapped, is_root=False, depth=unwrapped_depth + 1
            )
            if matched is not None:
                return matched
        return None

    if isinstance(unwrapped, dict) and len(unwrapped) == 1:
        k, v = next(iter(unwrapped.items()))
        child_node = descendant_map.get(k)
        if child_node is not None:
            matched = _match_node(
                child_node, v, is_root=False, depth=unwrapped_depth + 1
            )
            if matched is not None:
                return matched
        return _match_descendant(descendant_map, v, depth=unwrapped_depth + 1)

    return None


def _match_node(
    curr_node: _PathTrieNode,
    curr_val: Any,
    *,
    is_root: bool,
    depth: int = 0,
) -> tuple[_PathTrieNode, Any] | None:
    """Recursively match `curr_val` against `curr_node` with backtracking and depth bounds."""
    if depth > MAX_TRIE_DEPTH:
        return None

    if (
        curr_node.parser is not None
        and not curr_node.children
        and not curr_node.descendant_children
    ):
        return curr_node, curr_val

    if curr_node.children:
        if isinstance(curr_val, dict):
            if is_root or len(curr_val) == 1:
                if len(curr_val) == 1:
                    k, v = next(iter(curr_val.items()))
                    child_node = curr_node.children.get(k)
                    if child_node is not None:
                        matched = _match_node(
                            child_node, v, is_root=False, depth=depth + 1
                        )
                        if matched is not None:
                            return matched
            else:
                for k, child_node in curr_node.children.items():
                    if k in curr_val:
                        matched = _match_node(
                            child_node,
                            curr_val[k],
                            is_root=False,
                            depth=depth + 1,
                        )
                        if matched is not None:
                            return matched
        elif isinstance(curr_val, str):
            child_node = curr_node.children.get(curr_val)
            if child_node is not None:
                matched = _match_node(
                    child_node, curr_val, is_root=False, depth=depth + 1
                )
                if matched is not None:
                    return matched

    if curr_node.descendant_children:
        matched = _match_descendant(
            curr_node.descendant_children, curr_val, depth=depth + 1
        )
        if matched is not None:
            return matched

    if curr_node.parser is not None:
        return curr_node, curr_val
    return None


def _match_trie(
    root: _PathTrieNode, expr_json: Any
) -> tuple[_PathTrieNode, Any] | None:
    """Match `expr_json` against the compiled JSON Path trie."""
    if not isinstance(expr_json, dict) or len(expr_json) != 1:
        return None
    return _match_node(root, expr_json, is_root=True, depth=0)


def _invoke_matched_parser(
    node: _PathTrieNode, matched_val: Any, context_json: Any
) -> ParseRecord:
    """Invoke `node.parser` and normalize its return value into a `ParseRecord`."""
    parser = node.parser
    if parser is None:
        return UNSUPPORTED_RECORD

    result = (
        parser(matched_val, context_json)
        if node.accepts_expr_json
        else parser(matched_val)
    )
    if isinstance(result, Expr):
        return ParseRecord("Leaf", result, ())
    if isinstance(result, ParseRecord):
        return result
    return UNSUPPORTED_RECORD


def _ensure_builtin_parsers_registered() -> None:
    """Ensure all domain parser modules have registered their paths into `_DISPATCH_TRIE`."""
    global _BUILTIN_PARSERS_LOADED
    if not _BUILTIN_PARSERS_LOADED:
        from polars_bigquery.core.compiler import parser as _parser_pkg  # noqa: F401

        _BUILTIN_PARSERS_LOADED = True


def dispatch_parser(expr_json: Any) -> ParseRecord:
    """Dispatch a Polars expression JSON node to its matching registered parser.

    Returns `UNSUPPORTED_RECORD` (`("Leaf", Unsupported(), ())`) if no registered
    JSON Path matches `expr_json`.
    """
    _ensure_builtin_parsers_registered()
    matched = _match_trie(_DISPATCH_TRIE, expr_json)
    if matched is None:
        return UNSUPPORTED_RECORD
    node, matched_val = matched
    return _invoke_matched_parser(node, matched_val, expr_json)


def dispatch_literal_parser(literal_payload: Any) -> ParseRecord:
    """Dispatch a Polars literal payload directly against the `$.Literal` subtrie.

    Avoids allocating synthetic `{"Literal": elem}` wrapper dictionaries when
    compiling large list literals.
    """
    _ensure_builtin_parsers_registered()
    literal_root = _DISPATCH_TRIE.children.get("Literal")
    if literal_root is None:
        return UNSUPPORTED_RECORD
    matched = _match_node(literal_root, literal_payload, is_root=False, depth=1)
    if matched is None:
        return UNSUPPORTED_RECORD
    node, matched_val = matched
    return _invoke_matched_parser(node, matched_val, literal_payload)


def extract_binary_operands(expr_json: Any) -> tuple[Any, Any] | None:
    """Extract `(left, right)` operand JSONs from a Polars `BinaryExpr` JSON node."""
    if isinstance(expr_json, dict):
        binary_body = expr_json.get("BinaryExpr", expr_json)
        if isinstance(binary_body, dict):
            op_val = binary_body.get("op")
            if op_val is not None and not isinstance(op_val, str):
                return None
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
    return ParseRecord("Binary", binary_cls, operands)


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


def _has_valid_unit_function_spec(expr_json: Any) -> bool:
    """Verify that a unit `Function` node does not carry unrecognized option payloads."""
    if not isinstance(expr_json, dict):
        return True
    func_body = expr_json.get("Function", expr_json)
    if not isinstance(func_body, dict) or "function" not in func_body:
        return True
    fn_spec = func_body["function"]
    if isinstance(fn_spec, str):
        return True
    if isinstance(fn_spec, dict) and len(fn_spec) == 1:
        (inner_spec,) = fn_spec.values()
        if isinstance(inner_spec, str) or inner_spec is None:
            return True
        if isinstance(inner_spec, dict) and len(inner_spec) == 1:
            (leaf_spec,) = inner_spec.values()
            return leaf_spec is None or leaf_spec == {}
    return False


def parse_unary_function(
    expr_json: Any,
    unary_cls: type[UnaryExpr],
    *,
    exact_inputs: bool = True,
) -> ParseRecord:
    """Build a Unary `ParseRecord` for `unary_cls` from a `Function` JSON node."""
    if not _has_valid_unit_function_spec(expr_json):
        return UNSUPPORTED_RECORD
    inputs = extract_function_inputs(expr_json)
    if inputs is None:
        return UNSUPPORTED_RECORD
    if (exact_inputs and len(inputs) != 1) or len(inputs) < 1:
        return UNSUPPORTED_RECORD
    return ParseRecord("Unary", unary_cls, (inputs[0],))


def parse_binary_function(
    expr_json: Any,
    binary_cls: Any,
) -> ParseRecord:
    """Build a Binary `ParseRecord` for `binary_cls` from a 2-input `Function` JSON node."""
    if not _has_valid_unit_function_spec(expr_json):
        return UNSUPPORTED_RECORD
    inputs = extract_function_inputs(expr_json)
    if inputs is None or len(inputs) != 2:
        return UNSUPPORTED_RECORD
    return ParseRecord("Binary", binary_cls, (inputs[0], inputs[1]))


@register_parser("$.Literal..Null")
def parse_null_literal(_value: Any) -> NullLiteral:
    """Parse a Null value from Polars JSON into a NullLiteral."""
    return NullLiteral()


@register_parser("$.Column")
def parse_column_expr(column_json: Any) -> ParseRecord:
    """Extract Column leaf node for a Polars `Column` node."""
    if not isinstance(column_json, str):
        return UNSUPPORTED_RECORD
    return ParseRecord("Leaf", Column(name=column_json), ())
