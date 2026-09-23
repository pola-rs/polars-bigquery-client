from __future__ import annotations

import functools
from typing import Any

from polars_bigquery.core.compiler.ir.base import Expr, Unsupported
from polars_bigquery.core.compiler.ir.string import (
    Contains,
    EndsWith,
    Lowercase,
    StartsWith,
    StringLiteral,
    Uppercase,
)
from polars_bigquery.core.compiler.parser.base import (
    UNSUPPORTED_RECORD,
    ParseRecord,
    extract_bool_options,
    extract_function_inputs,
    parse_binary_function,
    parse_unary_function,
    register_parser,
)

STRING_TYPES = frozenset({"String", "StringOwned"})


@register_parser(*(f"$.Literal..{str_type}" for str_type in sorted(STRING_TYPES)))
def parse_string_literal(value: Any) -> Expr:
    """Parse a string value from Polars JSON into a StringLiteral or Unsupported."""
    if not isinstance(value, str):
        return Unsupported()
    return StringLiteral(value=value)


@register_parser("$.Function.function.StringExpr.Uppercase")
def parse_uppercase(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Uppercase` StringFunction node into an IR `Uppercase` record."""
    return parse_unary_function(expr_json, Uppercase, exact_inputs=True, func_spec=spec)


@register_parser("$.Function.function.StringExpr.Lowercase")
def parse_lowercase(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Lowercase` StringFunction node into an IR `Lowercase` record."""
    return parse_unary_function(expr_json, Lowercase, exact_inputs=True, func_spec=spec)


@register_parser("$.Function.function.StringExpr.StartsWith")
def parse_starts_with(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `StartsWith` StringFunction node into an IR `StartsWith` record."""
    return parse_binary_function(expr_json, StartsWith, func_spec=spec)


@register_parser("$.Function.function.StringExpr.EndsWith")
def parse_ends_with(spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `EndsWith` StringFunction node into an IR `EndsWith` record."""
    return parse_binary_function(expr_json, EndsWith, func_spec=spec)


@register_parser("$.Function.function.StringExpr.Contains")
def parse_contains(contains_spec: Any, expr_json: Any) -> ParseRecord:
    """Extract IR constructor and child JSONs for a Polars `Contains` StringFunction node."""
    inputs = extract_function_inputs(expr_json)
    if inputs is None or len(inputs) != 2:
        return UNSUPPORTED_RECORD
    opts = extract_bool_options(
        contains_spec,
        "Contains",
        {"literal": False, "strict": True},
    )
    if opts is None or (not opts["literal"] and not opts["strict"]):
        return UNSUPPORTED_RECORD
    return ParseRecord(
        "Binary",
        functools.partial(Contains, literal=opts["literal"]),
        (inputs[0], inputs[1]),
    )
