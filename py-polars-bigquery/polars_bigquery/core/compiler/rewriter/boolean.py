from __future__ import annotations

from polars_bigquery.core.compiler.ir.base import Expr
from polars_bigquery.core.compiler.ir.boolean import And, Not, Or
from polars_bigquery.core.compiler.rewriter.base import rewrite_tree


def rewrite_de_morgan_node(node: Expr) -> Expr:
    """Rewrite `Not(Or(left, right))` into `And(Not(left), Not(right))`.

    We intentionally only apply De Morgan's law in the direction that turns `Or`
    into `And` (and not `Not(And(...))` into `Or(...)`), because `And` allows
    either branch to contain an `Unsupported` descendant while still preserving
    predicate pushdown for the supported branch.
    """
    if isinstance(node, Not) and isinstance(node.expr, Or):
        return And(
            left=Not(expr=node.expr.left),
            right=Not(expr=node.expr.right),
        )
    return node


def rewrite_de_morgan(root: Expr) -> Expr:
    """Apply De Morgan's law across an IR tree to turn negated `Or` nodes into `And` nodes."""
    return rewrite_tree(root, rewrite_de_morgan_node)
