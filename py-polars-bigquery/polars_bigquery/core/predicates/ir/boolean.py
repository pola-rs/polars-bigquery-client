from __future__ import annotations

from dataclasses import dataclass

from polars_bigquery.core.predicates.ir.base import BinaryExpr, Literal, UnaryExpr


@dataclass(frozen=True)
class BoolLiteral(Literal):
    """IR node representing a boolean literal."""

    value: bool


@dataclass(frozen=True)
class And(BinaryExpr):
    """IR node representing logical conjunction (AND)."""


@dataclass(frozen=True)
class Or(BinaryExpr):
    """IR node representing logical disjunction (OR)."""


@dataclass(frozen=True)
class Not(UnaryExpr):
    """IR node representing logical negation (NOT)."""


@dataclass(frozen=True)
class IsNull(UnaryExpr):
    """IR node representing an IS NULL check."""


@dataclass(frozen=True)
class IsNotNull(UnaryExpr):
    """IR node representing an IS NOT NULL check."""


LOGICAL_BINARY_OPS: dict[str, type[BinaryExpr]] = {
    "And": And,
    "Or": Or,
}

BOOLEAN_UNARY_FUNCTIONS: dict[str, type[UnaryExpr]] = {
    "Not": Not,
    "IsNull": IsNull,
    "IsNotNull": IsNotNull,
}
