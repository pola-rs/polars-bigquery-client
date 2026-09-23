from __future__ import annotations

import collections.abc
import dataclasses
import inspect
import logging
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NamedTuple, TypeVar

from polars_bigquery.core.compiler.ir.base import (
    BinaryExpr,
    Column,
    Expr,
    NullLiteral,
    UnaryExpr,
    Unsupported,
)

logger = logging.getLogger(__name__)

RecordKind = Literal["Leaf", "Unary", "Binary", "Variadic"]


class ParseRecord(NamedTuple):
    """Strongly-typed AST parse record compatible with 3-tuple unpacking."""

    kind: RecordKind
    node: Any
    children: Sequence[Any]


ParserResult = Expr | ParseRecord
ParserFunc = Callable[..., ParserResult]
_F = TypeVar("_F", bound=ParserFunc)

UNSUPPORTED_RECORD: ParseRecord = ParseRecord("Leaf", Unsupported(), ())
MAX_TRIE_DEPTH = 16
RECOVERABLE_PARSE_ERRORS: tuple[type[Exception], ...] = (
    ArithmeticError,
    AttributeError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
)

_VALID_BINARY_EXPR_KEYS = frozenset({"left", "op", "right"})
_VALID_FUNCTION_EXPR_KEYS = frozenset({"input", "function"})


@dataclasses.dataclass(frozen=True, slots=True)
class _CompiledRoute:
    """Pre-compiled dispatch route bound to a terminal Trie node."""

    path: str
    parser: ParserFunc
    accepts_expr_json: bool


@dataclasses.dataclass(slots=True)
class _DispatchTrieNode:
    """Compiled JSON Path trie node supporting `.segment` and `..segment` transitions."""

    children: dict[str, _DispatchTrieNode] = dataclasses.field(default_factory=dict)
    descendants: dict[str, _DispatchTrieNode] = dataclasses.field(default_factory=dict)
    route: _CompiledRoute | None = None


# Registry mapping JSON Path expressions (e.g. "$.BinaryExpr.op.Eq",
# "$.Function.function.Boolean.Not", "$.Literal..Int64") to parser functions.
_PARSERS_REGISTRY: dict[str, ParserFunc] = {}
_PARSER_ACCEPTS_EXPR_JSON: dict[str, bool] = {}
_TRIE_ROOT = _DispatchTrieNode()
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

    if len(steps) > MAX_TRIE_DEPTH:
        msg = (
            f"Invalid JSON path {path!r}: depth {len(steps)} exceeds "
            f"MAX_TRIE_DEPTH ({MAX_TRIE_DEPTH})"
        )
        raise ValueError(msg)
    return tuple(steps)


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


def _insert_compiled_route(
    steps: tuple[tuple[str, str], ...], route: _CompiledRoute
) -> None:
    """Insert a validated JSON Path step sequence into `_TRIE_ROOT`."""
    curr_node = _TRIE_ROOT
    for op, segment in steps:
        table = curr_node.descendants if op == ".." else curr_node.children
        next_node = table.get(segment)
        if next_node is None:
            next_node = _DispatchTrieNode()
            table[segment] = next_node
        curr_node = next_node
    if curr_node.route is not None and curr_node.route.parser is not route.parser:
        msg = (
            f"Conflicting trie route for {route.path!r}: node already bound to "
            f"{curr_node.route.path!r} ({curr_node.route.parser!r})"
        )
        raise ValueError(msg)
    curr_node.route = route


def register_parser(*paths: str) -> Callable[[_F], _F]:
    """Decorator to register a parser function for one or more JSON Paths."""
    if not paths:
        msg = "register_parser requires at least one JSON Path string"
        raise ValueError(msg)

    parsed_paths = [(path, parse_json_path(path)) for path in paths]

    def _decorator(func: _F) -> _F:
        accepts_expr_json = _accepts_second_positional_arg(func)
        for path, _ in parsed_paths:
            existing = _PARSERS_REGISTRY.get(path)
            if existing is not None and existing is not func:
                msg = (
                    f"Duplicate parser registration for JSON path {path!r}: "
                    f"{existing!r} vs {func!r}"
                )
                raise ValueError(msg)
        for path, steps in parsed_paths:
            _PARSERS_REGISTRY[path] = func
            _PARSER_ACCEPTS_EXPR_JSON[path] = accepts_expr_json
            _insert_compiled_route(
                steps,
                _CompiledRoute(
                    path=path,
                    parser=func,
                    accepts_expr_json=accepts_expr_json,
                ),
            )
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


def _extract_enum_tag_and_payload(val: Any) -> tuple[str, Any] | None:
    """Extract `(tag, payload)` from a unit string or single-key enum dict."""
    if isinstance(val, str):
        return val, None
    if isinstance(val, dict) and len(val) == 1:
        return next(iter(val.items()))
    return None


def _step_descendant(
    node: _DispatchTrieNode, payload: Any, depth: int
) -> tuple[_DispatchTrieNode, str, Any, int] | None:
    """Resolve a `..segment` transition across transparent literal wrappers."""
    curr, curr_depth = _unwrap_literal_with_depth(payload, depth)
    if curr_depth > MAX_TRIE_DEPTH:
        return None
    tag_and_val = _extract_enum_tag_and_payload(curr)
    if tag_and_val is None:
        return None
    tag, val = tag_and_val
    desc_node = node.descendants.get(tag)
    if desc_node is None:
        return None
    return desc_node, tag, val, curr_depth + 1


def _step_child(
    node: _DispatchTrieNode, curr_val: Any
) -> tuple[_DispatchTrieNode, str, Any] | None:
    """Resolve a direct `.segment` transition on `node.children`."""
    if not node.children:
        return None
    tag_and_payload = _extract_enum_tag_and_payload(curr_val)
    if tag_and_payload is not None:
        tag, payload = tag_and_payload
        cand_node = node.children.get(tag)
        if cand_node is not None:
            return cand_node, tag, payload
        return None
    if isinstance(curr_val, dict) and len(curr_val) > 1:
        matched_keys = curr_val.keys() & node.children.keys()
        if len(matched_keys) == 1:
            tag = next(iter(matched_keys))
            return node.children[tag], tag, curr_val[tag]
    return None


def _match_trie_route(expr_json: Any) -> tuple[_CompiledRoute, Any] | None:
    """Walk `_TRIE_ROOT` against `expr_json` without runtime string formatting."""
    if not isinstance(expr_json, dict) or len(expr_json) != 1:
        return None

    curr_node = _TRIE_ROOT
    curr_val: Any = expr_json
    depth = 0

    while depth <= MAX_TRIE_DEPTH:
        next_node: _DispatchTrieNode | None = None
        tag: str = ""
        payload: Any = None

        child_step = _step_child(curr_node, curr_val)
        if child_step is not None:
            next_node, tag, payload = child_step
            depth += 1
        elif curr_node.descendants:
            desc_step = _step_descendant(curr_node, curr_val, depth)
            if desc_step is not None:
                next_node, tag, payload, depth = desc_step

        if next_node is None or depth > MAX_TRIE_DEPTH + 1:
            return None

        if next_node.route is not None and (
            payload is None or not (next_node.children or next_node.descendants)
        ):
            matched_val = (
                tag if (payload is None and isinstance(curr_val, str)) else payload
            )
            return next_node.route, matched_val

        curr_node = next_node
        curr_val = payload

    return None


def _validate_parse_record(record: ParseRecord) -> ParseRecord:
    """Validate structural invariants of a `ParseRecord` at the parser dispatch boundary."""
    kind, node, children = record
    if not isinstance(children, collections.abc.Sequence) or isinstance(
        children, (str, bytes, bytearray)
    ):
        return UNSUPPORTED_RECORD

    if kind == "Leaf":
        if isinstance(node, Expr) and len(children) == 0:
            return record
        return UNSUPPORTED_RECORD
    if kind == "Unary":
        if callable(node) and len(children) == 1:
            return record
        return UNSUPPORTED_RECORD
    if kind == "Binary":
        if callable(node) and len(children) == 2:
            return record
        return UNSUPPORTED_RECORD
    if kind == "Variadic":
        if callable(node) and len(children) >= 1:
            return record
        return UNSUPPORTED_RECORD
    return UNSUPPORTED_RECORD


def _invoke_matched_parser(
    parser: ParserFunc | None,
    matched_val: Any,
    context_json: Any,
    *,
    accepts_expr_json: bool = False,
) -> ParseRecord:
    """Invoke `parser` within an exception boundary and normalize into a validated `ParseRecord`."""
    if parser is None:
        return UNSUPPORTED_RECORD

    try:
        result = (
            parser(matched_val, context_json)
            if accepts_expr_json
            else parser(matched_val)
        )
    except RECOVERABLE_PARSE_ERRORS:
        logger.debug(
            "Parser %r raised an exception; degrading node to UNSUPPORTED_RECORD",
            parser,
            exc_info=True,
        )
        return UNSUPPORTED_RECORD

    if isinstance(result, Expr):
        return ParseRecord("Leaf", result, ())
    if isinstance(result, ParseRecord):
        return _validate_parse_record(result)
    return UNSUPPORTED_RECORD


def _ensure_builtin_parsers_registered() -> None:
    """Ensure all domain parser modules have registered their paths into `_PARSERS_REGISTRY`."""
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
    matched = _match_trie_route(expr_json)
    if matched is None:
        return UNSUPPORTED_RECORD
    route, matched_val = matched
    return _invoke_matched_parser(
        route.parser,
        matched_val,
        expr_json,
        accepts_expr_json=route.accepts_expr_json,
    )


def _is_valid_unit_spec(spec: Any) -> bool:
    """Return True if `spec` represents an unparameterized unit variant payload."""
    return spec is None or isinstance(spec, str)


def extract_binary_operands(expr_json: Any) -> tuple[Any, Any] | None:
    """Extract `(left, right)` operand JSONs from a Polars `BinaryExpr` JSON node."""
    if isinstance(expr_json, dict):
        if "BinaryExpr" in expr_json:
            if len(expr_json) != 1:
                return None
            binary_body = expr_json["BinaryExpr"]
        else:
            binary_body = expr_json
        if (
            isinstance(binary_body, dict)
            and binary_body.keys() <= _VALID_BINARY_EXPR_KEYS
        ):
            op_val = binary_body.get("op")
            if op_val is not None and not isinstance(op_val, str):
                return None
            left_json = binary_body.get("left")
            right_json = binary_body.get("right")
            if left_json is not None and right_json is not None:
                return left_json, right_json
    return None


def parse_binary_op(
    expr_json: Any,
    binary_cls: type[BinaryExpr],
    *,
    op_spec: Any = None,
) -> ParseRecord:
    """Build a Binary `ParseRecord` for `binary_cls` from a `BinaryExpr` JSON node."""
    if not _is_valid_unit_spec(op_spec):
        return UNSUPPORTED_RECORD
    operands = extract_binary_operands(expr_json)
    if operands is None:
        return UNSUPPORTED_RECORD
    return ParseRecord("Binary", binary_cls, operands)


def _has_valid_function_envelope(
    expr_json: Any, *, require_unit_leaf: bool = False
) -> bool:
    """Verify that a `Function` JSON envelope contains only valid keys and single-key enum tags."""
    if not isinstance(expr_json, dict):
        return True
    if "Function" in expr_json:
        if len(expr_json) != 1:
            return False
        func_body = expr_json["Function"]
    else:
        func_body = expr_json
    if not isinstance(func_body, dict) or not (
        func_body.keys() <= _VALID_FUNCTION_EXPR_KEYS
    ):
        return False
    if "function" not in func_body:
        return True

    curr_spec: Any = func_body["function"]
    if isinstance(curr_spec, str):
        return True
    for _ in range(MAX_TRIE_DEPTH):
        if not isinstance(curr_spec, dict) or len(curr_spec) != 1:
            return False
        (curr_spec,) = curr_spec.values()
        if curr_spec is None or isinstance(curr_spec, str):
            return True
        if (
            isinstance(curr_spec, dict)
            and not require_unit_leaf
            and all(isinstance(v, bool) for v in curr_spec.values())
        ):
            return True
    return False


def extract_bool_options(
    spec: Any,
    variant_name: str,
    defaults: Mapping[str, bool],
    *,
    allow_unit_or_bool_for: str | None = None,
) -> dict[str, bool] | None:
    """Unwrap and validate a boolean options dictionary for a parameterized function variant."""
    opts = (
        spec[variant_name]
        if isinstance(spec, dict) and len(spec) == 1 and variant_name in spec
        else spec
    )
    if isinstance(opts, dict):
        if not (opts.keys() <= defaults.keys()):
            return None
        resolved: dict[str, bool] = {}
        for key, default_val in defaults.items():
            raw_val = opts.get(key, default_val)
            if not isinstance(raw_val, bool):
                return None
            resolved[key] = raw_val
        return resolved
    if allow_unit_or_bool_for is not None:
        if isinstance(opts, bool):
            return {**defaults, allow_unit_or_bool_for: opts}
        if opts is None or opts == variant_name:
            return dict(defaults)
    return None


def extract_function_inputs(
    expr_json: Any, *, require_unit_leaf: bool = False
) -> list[Any] | None:
    """Extract the `input` list from a Polars `Function` JSON node or raw input list."""
    if isinstance(expr_json, list):
        return expr_json
    if not _has_valid_function_envelope(expr_json, require_unit_leaf=require_unit_leaf):
        return None
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
    exact_inputs: bool = True,
    func_spec: Any = None,
) -> ParseRecord:
    """Build a Unary `ParseRecord` for `unary_cls` from a `Function` JSON node."""
    if not _is_valid_unit_spec(func_spec):
        return UNSUPPORTED_RECORD
    inputs = extract_function_inputs(expr_json, require_unit_leaf=True)
    if inputs is None:
        return UNSUPPORTED_RECORD
    if (exact_inputs and len(inputs) != 1) or len(inputs) < 1:
        return UNSUPPORTED_RECORD
    return ParseRecord("Unary", unary_cls, (inputs[0],))


def parse_binary_function(
    expr_json: Any,
    binary_cls: Any,
    *,
    func_spec: Any = None,
) -> ParseRecord:
    """Build a Binary `ParseRecord` for `binary_cls` from a 2-input `Function` JSON node."""
    if not _is_valid_unit_spec(func_spec):
        return UNSUPPORTED_RECORD
    inputs = extract_function_inputs(expr_json, require_unit_leaf=True)
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
