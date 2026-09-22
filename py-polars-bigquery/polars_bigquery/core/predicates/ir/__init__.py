from __future__ import annotations

import collections
from collections.abc import Callable
from typing import Any

from polars_bigquery.core.predicates.ir.base import (
    NULL_LITERAL_PARSERS,
    BinaryExpr,
    Column,
    Expr,
    Literal,
    NullLiteral,
    UnaryExpr,
    Unsupported,
)
from polars_bigquery.core.predicates.ir.boolean import (
    BOOLEAN_LITERAL_PARSERS,
    BOOLEAN_UNARY_OPS,
    LOGICAL_BINARY_OPS,
    And,
    BoolLiteral,
    IsNotNull,
    IsNull,
    Not,
    Or,
)
from polars_bigquery.core.predicates.ir.comparison import (
    COMPARISON_BINARY_OPS,
    Eq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    NotEq,
)
from polars_bigquery.core.predicates.ir.numeric import (
    NUMERIC_LITERAL_PARSERS,
    NUMERIC_UNARY_OPS,
    FloatLiteral,
    IntLiteral,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
)
from polars_bigquery.core.predicates.ir.string import (
    STRING_BINARY_OPS,
    STRING_LITERAL_PARSERS,
    EndsWith,
    StartsWith,
    StringLiteral,
)
from polars_bigquery.core.predicates.ir.temporal import (
    TEMPORAL_LITERAL_PARSERS,
    DateLiteral,
    TimestampLiteral,
    TimestampUnit,
)

__all__ = [
    "And",
    "BinaryExpr",
    "BoolLiteral",
    "Column",
    "DateLiteral",
    "EndsWith",
    "Eq",
    "Expr",
    "FloatLiteral",
    "Gt",
    "GtEq",
    "IntLiteral",
    "IsFinite",
    "IsInfinite",
    "IsNan",
    "IsNotNan",
    "IsNotNull",
    "IsNull",
    "Literal",
    "Lt",
    "LtEq",
    "Not",
    "NotEq",
    "NullLiteral",
    "Or",
    "StartsWith",
    "StringLiteral",
    "TimestampLiteral",
    "TimestampUnit",
    "UnaryExpr",
    "Unsupported",
]

