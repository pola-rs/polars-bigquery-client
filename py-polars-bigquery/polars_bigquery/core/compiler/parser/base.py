from __future__ import annotations

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

RecordKind = Literal["Leaf", "Unary", "Binary", "Variadic"]


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
_PARSER_ACCEPTS_EXPR_JSON: dict[str, bool] = {}
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


def register_parser(*paths: str) -> Callable[[_F], _F]:
    """Decorator to register a parser function for one or more JSON Paths."""
    if not paths:
        msg = "register_parser requires at least one JSON Path string"
        raise ValueError(msg)

    for path in paths:
        parse_json_path(path)

    def _decorator(func: _F) -> _F:
        accepts_expr_json = _accepts_second_positional_arg(func)
        for path in paths:
            existing = _PARSERS_REGISTRY.get(path)
            if existing is not None and existing is not func:
                msg = (
                    f"Duplicate parser registration for JSON path {path!r}: "
                    f"{existing!r} vs {func!r}"
                )
                raise ValueError(msg)
        for path in paths:
            _PARSERS_REGISTRY[path] = func
            _PARSER_ACCEPTS_EXPR_JSON[path] = accepts_expr_json
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


def _resolve_literal_path(literal_body: Any) -> tuple[str, Any] | None:
    """Resolve a `Literal` payload across wrapper envelopes within `MAX_TRIE_DEPTH`."""
    curr = literal_body
    depth = 1
    while depth <= MAX_TRIE_DEPTH:
        curr, depth = _unwrap_literal_with_depth(curr, depth)
        if depth > MAX_TRIE_DEPTH:
            return None
        tag_and_val = _extract_enum_tag_and_payload(curr)
        if tag_and_val is None:
            return None
        lit_tag, lit_val = tag_and_val
        candidate_path = f"$.Literal..{lit_tag}"
        if candidate_path in _PARSERS_REGISTRY:
            return candidate_path, lit_val
        if lit_val is None:
            return None
        curr = lit_val
        depth += 1
    return None


def _resolve_json_path(expr_json: Any) -> tuple[str, Any] | None:
    """Resolve a Polars expression JSON node to its canonical `(json_path, matched_val)`."""
    root = _extract_enum_tag_and_payload(expr_json)
    if root is None:
        return None
    root_tag, body = root

    if root_tag == "Column":
        return "$.Column", body

    if root_tag == "Literal":
        return _resolve_literal_path(body)

    if root_tag == "BinaryExpr":
        if not isinstance(body, dict):
            return None
        op = _extract_enum_tag_and_payload(body.get("op"))
        if op is None:
            return None
        op_tag, op_spec = op
        return f"$.BinaryExpr.op.{op_tag}", op_spec

    if root_tag == "Function":
        if not isinstance(body, dict):
            return None
        fn = _extract_enum_tag_and_payload(body.get("function"))
        if fn is None:
            return None
        domain_tag, domain_body = fn
        if domain_body is None:
            return f"$.Function.function.{domain_tag}", domain_tag
        sub = _extract_enum_tag_and_payload(domain_body)
        if sub is None:
            return None
        func_tag, func_spec = sub
        return f"$.Function.function.{domain_tag}.{func_tag}", func_spec

    return f"$.{root_tag}", body


def _invoke_matched_parser(
    parser: ParserFunc | None,
    matched_val: Any,
    context_json: Any,
    *,
    accepts_expr_json: bool = False,
) -> ParseRecord:
    """Invoke `parser` within an exception boundary and normalize into a `ParseRecord`."""
    if parser is None:
        return UNSUPPORTED_RECORD

    try:
        result = (
            parser(matched_val, context_json)
            if accepts_expr_json
            else parser(matched_val)
        )
    except (
        ArithmeticError,
        LookupError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return UNSUPPORTED_RECORD

    if isinstance(result, Expr):
        return ParseRecord("Leaf", result, ())
    if isinstance(result, ParseRecord):
        return result
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
    resolved = _resolve_json_path(expr_json)
    if resolved is None:
        return UNSUPPORTED_RECORD
    path, matched_val = resolved
    parser = _PARSERS_REGISTRY.get(path)
    if parser is None:
        return UNSUPPORTED_RECORD
    return _invoke_matched_parser(
        parser,
        matched_val,
        expr_json,
        accepts_expr_json=_PARSER_ACCEPTS_EXPR_JSON.get(path, False),
    )


def _is_valid_unit_spec(spec: Any) -> bool:
    """Return True if `spec` represents an unparameterized unit variant payload."""
    return spec is None or isinstance(spec, str)


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
    """Verify that a `Function` JSON envelope contains only single-key enum tags."""
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
            if not require_unit_leaf:
                return True
            (leaf_spec,) = inner_spec.values()
            return leaf_spec is None or isinstance(leaf_spec, str)
    return False


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
