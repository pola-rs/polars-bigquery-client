from __future__ import annotations

import dataclasses

from polars_bigquery.core.compiler.ir.base import (
    Literal,
    UnaryExpr,
)


@dataclasses.dataclass(frozen=True)
class IntLiteral(Literal):
    """IR node representing an integer literal."""

    value: int


@dataclasses.dataclass(frozen=True)
class FloatLiteral(Literal):
    """IR node representing a floating-point literal."""

    value: float


@dataclasses.dataclass(frozen=True)
class IsNan(UnaryExpr):
    """IR node representing an IS_NAN check on a numeric expression."""


@dataclasses.dataclass(frozen=True)
class IsNotNan(UnaryExpr):
    """IR node representing a NOT IS_NAN check on a numeric expression."""


@dataclasses.dataclass(frozen=True)
class IsInfinite(UnaryExpr):
    """IR node representing an IS_INF check on a numeric expression."""


@dataclasses.dataclass(frozen=True)
class IsFinite(UnaryExpr):
    """IR node representing a finite check (NOT IS_INF AND NOT IS_NAN) on a numeric expression."""
