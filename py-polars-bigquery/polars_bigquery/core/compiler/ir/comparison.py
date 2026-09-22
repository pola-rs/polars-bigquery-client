from __future__ import annotations

import dataclasses

from polars_bigquery.core.compiler.ir.base import BinaryExpr


@dataclasses.dataclass(frozen=True)
class Eq(BinaryExpr):
    """IR node representing equality comparison (=)."""


@dataclasses.dataclass(frozen=True)
class NotEq(BinaryExpr):
    """IR node representing inequality comparison (!=)."""


@dataclasses.dataclass(frozen=True)
class Gt(BinaryExpr):
    """IR node representing greater-than comparison (>)."""


@dataclasses.dataclass(frozen=True)
class GtEq(BinaryExpr):
    """IR node representing greater-than-or-equal comparison (>=)."""


@dataclasses.dataclass(frozen=True)
class Lt(BinaryExpr):
    """IR node representing less-than comparison (<)."""


@dataclasses.dataclass(frozen=True)
class LtEq(BinaryExpr):
    """IR node representing less-than-or-equal comparison (<=)."""


@dataclasses.dataclass(frozen=True)
class IsIn(BinaryExpr):
    """IR node representing set membership comparison (IN)."""
