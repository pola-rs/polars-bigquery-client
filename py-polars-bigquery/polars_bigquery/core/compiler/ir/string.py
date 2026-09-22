from __future__ import annotations

import dataclasses

from polars_bigquery.core.compiler.ir.base import (
    BinaryExpr,
    Expr,
    Literal,
    UnaryExpr,
)


@dataclasses.dataclass(frozen=True)
class StringLiteral(Literal):
    """IR node representing a string literal."""

    value: str


@dataclasses.dataclass(frozen=True)
class Uppercase(UnaryExpr):
    """IR node representing a string UPPER conversion."""


@dataclasses.dataclass(frozen=True)
class Lowercase(UnaryExpr):
    """IR node representing a string LOWER conversion."""


@dataclasses.dataclass(frozen=True)
class Contains(BinaryExpr):
    """IR node representing a string CONTAINS check (regex or literal substring)."""

    literal: bool = False

    @property
    def expr(self) -> Expr:
        return self.left

    @property
    def pattern(self) -> Expr:
        return self.right


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
