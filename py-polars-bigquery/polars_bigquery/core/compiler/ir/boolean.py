from __future__ import annotations

import dataclasses

from polars_bigquery.core.compiler.ir.base import (
    BinaryExpr,
    Literal,
    UnaryExpr,
)


@dataclasses.dataclass(frozen=True)
class BoolLiteral(Literal):
    """IR node representing a boolean literal."""

    value: bool


@dataclasses.dataclass(frozen=True)
class And(BinaryExpr):
    """IR node representing logical conjunction (AND)."""


@dataclasses.dataclass(frozen=True)
class Or(BinaryExpr):
    """IR node representing logical disjunction (OR)."""


@dataclasses.dataclass(frozen=True)
class Not(UnaryExpr):
    """IR node representing logical negation (NOT)."""


@dataclasses.dataclass(frozen=True)
class IsNull(UnaryExpr):
    """IR node representing an IS NULL check."""


@dataclasses.dataclass(frozen=True)
class IsNotNull(UnaryExpr):
    """IR node representing an IS NOT NULL check."""
