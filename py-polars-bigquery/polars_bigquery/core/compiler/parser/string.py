from __future__ import annotations

import functools
from typing import Any

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
    extract_function_inputs,
    parse_binary_function,
    parse_unary_function,
    register_parser,
)

STRING_TYPES = frozenset({"String", "StringOwned"})


@register_parser(*(f"$.Literal..{str_type}" for str_type in sorted(STRING_TYPES)))
def parse_string_literal(value: Any) -> StringLiteral:
    """Parse a string value from Polars JSON into a StringLiteral."""
    return StringLiteral(value=str(value))


@register_parser("$.Function.function.StringExpr.Uppercase")
def parse_uppercase(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Uppercase` StringFunction node into an IR `Uppercase` record."""
    return parse_unary_function(expr_json, Uppercase, exact_inputs=True)


@register_parser("$.Function.function.StringExpr.Lowercase")
def parse_lowercase(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `Lowercase` StringFunction node into an IR `Lowercase` record."""
    return parse_unary_function(expr_json, Lowercase, exact_inputs=True)


@register_parser("$.Function.function.StringExpr.StartsWith")
def parse_starts_with(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `StartsWith` StringFunction node into an IR `StartsWith` record."""
    return parse_binary_function(expr_json, StartsWith)


@register_parser("$.Function.function.StringExpr.EndsWith")
def parse_ends_with(_spec: Any, expr_json: Any) -> ParseRecord:
    """Parse a Polars `EndsWith` StringFunction node into an IR `EndsWith` record."""
    return parse_binary_function(expr_json, EndsWith)


@register_parser("$.Function.function.StringExpr.Contains")
def parse_contains(contains_spec: Any, expr_json: Any) -> ParseRecord:
    """Extract IR constructor and child JSONs for a Polars `Contains` StringFunction node."""
    inputs = extract_function_inputs(expr_json)
    if inputs is None or len(inputs) != 2:
        return UNSUPPORTED_RECORD
    contains_opts = (
        contains_spec["Contains"]
        if isinstance(contains_spec, dict) and "Contains" in contains_spec
        else contains_spec
    )
    if isinstance(contains_opts, dict):
        literal = bool(contains_opts.get("literal", False))
        return (
            "Binary",
            functools.partial(Contains, literal=literal),
            (inputs[0], inputs[1]),
        )
    return UNSUPPORTED_RECORD
