from __future__ import annotations

from polars_bigquery.core.predicates.ir.base import BinaryExpr
from polars_bigquery.core.predicates.ir.comparison import (
    Eq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    NotEq,
)

# Keys represent Polars Rust AST `Operator` enum variant names emitted under
# `{"BinaryExpr": {"op": "<key>"}}` when serializing `pl.Expr.meta.serialize(format="json")`.
COMPARISON_BINARY_OPS: dict[str, type[BinaryExpr]] = {
    "Eq": Eq,
    "NotEq": NotEq,
    "Gt": Gt,
    "GtEq": GtEq,
    "Lt": Lt,
    "LtEq": LtEq,
}
