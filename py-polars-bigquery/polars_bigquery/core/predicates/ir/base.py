from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class Expr:
    """Base class for all predicate Intermediate Representation (IR) nodes."""

    def children(self) -> tuple[Expr, ...]:
        """Return the child expressions of this node in order."""
        return ()


@dataclasses.dataclass(frozen=True)
class UnaryExpr(Expr):
    """Base class for unary operation IR nodes."""

    expr: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.expr,)


@dataclasses.dataclass(frozen=True)
class BinaryExpr(Expr):
    """Base class for binary operation IR nodes."""

    left: Expr
    right: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)


@dataclasses.dataclass(frozen=True)
class Unsupported(Expr):
    """IR node representing an unsupported or unknown expression."""

    operands: tuple[Expr, ...] = ()

    def children(self) -> tuple[Expr, ...]:
        return self.operands


@dataclasses.dataclass(frozen=True)
class Column(Expr):
    """IR node representing a column reference."""

    name: str


@dataclasses.dataclass(frozen=True)
class Literal(Expr):
    """Base class for literal IR nodes."""


@dataclasses.dataclass(frozen=True)
class NullLiteral(Literal):
    """IR node representing a NULL literal."""


def parse_null_literal(_value: Any) -> NullLiteral:
    """Parse a Null value from Polars JSON into a NullLiteral."""
    return NullLiteral()


# Keys represent Polars Rust AST `LiteralValue` / `AnyValue` enum variant names
# emitted inside `{"Literal": ...}` in serialized Polars JSON.
NULL_LITERAL_PARSERS: dict[str, Callable[[Any], Expr]] = {
    "Null": parse_null_literal,
}
