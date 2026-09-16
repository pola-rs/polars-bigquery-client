from __future__ import annotations

from dataclasses import dataclass

from polars_bigquery.core.predicates.ir.base import BinaryExpr


@dataclass(frozen=True)
class Eq(BinaryExpr):
    """IR node representing equality comparison (=)."""


@dataclass(frozen=True)
class NotEq(BinaryExpr):
    """IR node representing inequality comparison (!=)."""


@dataclass(frozen=True)
class Gt(BinaryExpr):
    """IR node representing greater-than comparison (>)."""


@dataclass(frozen=True)
class GtEq(BinaryExpr):
    """IR node representing greater-than-or-equal comparison (>=)."""


@dataclass(frozen=True)
class Lt(BinaryExpr):
    """IR node representing less-than comparison (<)."""


@dataclass(frozen=True)
class LtEq(BinaryExpr):
    """IR node representing less-than-or-equal comparison (<=)."""


COMPARISON_BINARY_OPS: dict[str, type[BinaryExpr]] = {
    "Eq": Eq,
    "NotEq": NotEq,
    "Gt": Gt,
    "GtEq": GtEq,
    "Lt": Lt,
    "LtEq": LtEq,
}
