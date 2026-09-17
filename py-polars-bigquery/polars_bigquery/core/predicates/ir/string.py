from __future__ import annotations

import dataclasses
from typing import Any

from polars_bigquery.core.predicates.ir.base import BinaryExpr, Expr, Literal

STRING_TYPES = frozenset({"String", "StringOwned"})


@dataclasses.dataclass(frozen=True)
class StringLiteral(Literal):
    """IR node representing a string literal."""

    value: str


@dataclasses.dataclass(frozen=True)
class StartsWith(BinaryExpr):
    """IR node representing a string STARTS_WITH check."""

    @property
    def expr(self) -> Expr:
        return self.left

    @property
    def prefix(self) -> Expr:
        return self.right


@dataclasses.dataclass(frozen=True)
class EndsWith(BinaryExpr):
    """IR node representing a string ENDS_WITH check."""

    @property
    def expr(self) -> Expr:
        return self.left

    @property
    def suffix(self) -> Expr:
        return self.right


# Keys represent Polars Rust AST `StringFunction` enum variant names emitted under
# `{"Function": {"function": {"StringExpr": "<key>"}}}` in serialized Polars JSON.
STRING_BINARY_FUNCTIONS: dict[str, type[BinaryExpr]] = {
    "StartsWith": StartsWith,
    "EndsWith": EndsWith,
}


def parse_string_literal(value: Any) -> StringLiteral:
    """Parse a string value from Polars JSON into a StringLiteral."""
    return StringLiteral(value=str(value))
