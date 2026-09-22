from __future__ import annotations

import dataclasses

from polars_bigquery.core.compiler.ir.base import BinaryExpr, Literal


@dataclasses.dataclass(frozen=True)
class ListLiteral(Literal):
    """IR node representing a homogeneous list of scalar literals (e.g. for IN)."""

    values: tuple[Literal, ...]


@dataclasses.dataclass(frozen=True)
class IsIn(BinaryExpr):
    """IR node representing set membership comparison (IN)."""
