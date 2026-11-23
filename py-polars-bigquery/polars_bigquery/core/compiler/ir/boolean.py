from __future__ import annotations

import dataclasses

from polars_bigquery.core.compiler.ir.base import (
    BinaryExpr,
    Literal,
    TernaryExpr,
    UnaryExpr,
    VariadicExpr,
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
    """IR node representing an IS NOT NULL check.

    In addition to `Expr.is_not_null()`, Polars lowers `LazyFrame.drop_nulls()`
    (outside of a `.filter()`) into `is_not_null()` predicate checks, which are
    pushed down to BigQuery using this IR node.

    By contrast, `Expr.drop_nulls()` (`pl.col(...).drop_nulls()`) shrinks a
    `Series` length and broadcasts length-1 results across null rows, so it
    intentionally has no IR node and degrades to `Unsupported`.
    """


@dataclasses.dataclass(frozen=True)
class Coalesce(VariadicExpr):
    """IR node representing a COALESCE expression."""


@dataclasses.dataclass(frozen=True)
class FillNull(BinaryExpr):
    """IR node representing an IFNULL expression."""


@dataclasses.dataclass(frozen=True)
class Ternary(TernaryExpr):
    """IR node representing a single 3-operand conditional expression (IF)."""


@dataclasses.dataclass(frozen=True)
class When(VariadicExpr):
    """IR node representing a multi-clause conditional expression (CASE WHEN).

    `operands` stores one or more `(predicate, truthy)` pairs followed by the
    trailing `falsy` (`otherwise`) expression:
    `(predicate_1, truthy_1, ..., predicate_n, truthy_n, falsy)`.
    """
