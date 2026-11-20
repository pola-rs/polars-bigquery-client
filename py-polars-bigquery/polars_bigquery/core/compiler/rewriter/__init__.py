"""Functions for rewriting an Expr IR tree before SQL generation."""

from __future__ import annotations

from polars_bigquery.core.compiler.ir.base import Expr
from polars_bigquery.core.compiler.rewriter import base, boolean
from polars_bigquery.core.compiler.rewriter.base import RewriterFunc
from polars_bigquery.core.compiler.rewriter.boolean import (
    rewrite_boolean_literals,
    rewrite_boolean_when,
    rewrite_de_morgan,
    rewrite_when,
)

_REWRITERS: tuple[RewriterFunc, ...] = (
    # `rewrite_drop_nulls` is intentionally excluded from the pushdown pipeline:
    # Polars `drop_nulls()` changes Series length and broadcasts length-1 results
    # across the entire DataFrame (including null rows), so hoisting `IS NOT NULL`
    # to the root can drop matching rows and suppress Polars `ShapeError`.
    rewrite_boolean_when,
    rewrite_boolean_literals,
    rewrite_when,
    # Apply rewrite_de_morgan last in case any of the above add an OR.
    rewrite_de_morgan,
)


def rewrite_ir(root: Expr) -> Expr:
    """Apply all registered IR rewriters to `root` in sequence."""
    curr = root
    for rewriter in _REWRITERS:
        curr = rewriter(curr)
    return curr


__all__ = [
    "base",
    "boolean",
    "rewrite_ir",
]
