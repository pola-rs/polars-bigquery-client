"""Immutable logical scan plan IR shared across BigQuery backend executors."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Sequence
from typing import Literal

import arrow_bigquery
import polars as pl

ArrowBufferCompression = Literal["lz4frame", "zstd", "unspecified"]
MAX_BIGQUERY_INT64 = (1 << 63) - 1


def validate_maximum_bytes_billed(value: object) -> int | None:
    """Validate optional `maximum_bytes_billed` (`0 <= value <= (1 << 63) - 1`).

    Avoids formatting arbitrarily huge Python `int` values into strings so
    inputs like `10**5000` cannot trip CPython's `int_max_str_digits` limit.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = (
            f"maximum_bytes_billed must be an int or None, got {type(value).__name__}."
        )
        raise TypeError(msg)
    if value < 0:
        msg = "maximum_bytes_billed must be non-negative."
        raise ValueError(msg)
    if value > MAX_BIGQUERY_INT64:
        msg = f"maximum_bytes_billed must be <= {MAX_BIGQUERY_INT64}."
        raise ValueError(msg)
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class PlanNode(abc.ABC):
    """Base class for all logical BigQuery scan plan nodes.

    Note that `PlanNode` instances containing mutable or expression Polars
    objects (`pl.Schema` in `TableScan` or `pl.Expr` in `Filter`, and any
    enclosing tree) are unhashable.
    """

    @abc.abstractmethod
    def children(self) -> tuple[PlanNode, ...]:
        """Return the direct child plan nodes of this node in order."""


@dataclasses.dataclass(frozen=True, slots=True)
class QueryScan(PlanNode):
    """Leaf plan node representing a GoogleSQL `SELECT` query scan."""

    raw_query: str
    columns: tuple[str, ...]
    maintain_order: bool = True
    arrow_buffer_compression: ArrowBufferCompression = "lz4frame"
    use_query_cache: bool | None = None
    explicit_schema: bool = False
    maximum_bytes_billed: int | None = None

    def __post_init__(self) -> None:
        if len(self.columns) == 0:
            msg = "QueryScan requires at least one column."
            raise ValueError(msg)
        if len(set(self.columns)) != len(self.columns):
            msg = "QueryScan columns must be unique."
            raise ValueError(msg)
        validate_maximum_bytes_billed(self.maximum_bytes_billed)

    def children(self) -> tuple[PlanNode, ...]:
        return ()


@dataclasses.dataclass(frozen=True, slots=True)
class TableScan(PlanNode):
    """Leaf plan node representing a BigQuery Storage Read API table scan.

    Instances are unhashable because `pl.Schema` inherits from `OrderedDict`.
    """

    table_ref: arrow_bigquery.BigQueryTableId
    schema: pl.Schema
    maintain_order: bool = False

    def __post_init__(self) -> None:
        # Copy the caller-provided schema via pl.Schema(...) so external
        # mutation of the OrderedDict subclass cannot mutate this node and
        # unspecified dtype classes fail eagerly with TypeError.
        copied_schema = pl.Schema(self.schema)
        if len(copied_schema) == 0:
            msg = "TableScan requires a non-empty schema."
            raise ValueError(msg)
        object.__setattr__(self, "schema", copied_schema)

    @property
    def columns(self) -> tuple[str, ...]:
        """Column names of the table schema in declaration order."""
        return tuple(self.schema.keys())

    def children(self) -> tuple[PlanNode, ...]:
        return ()


@dataclasses.dataclass(frozen=True, slots=True)
class Filter(PlanNode):
    """Unary plan node representing a row filter predicate.

    Instances are unhashable because `__eq__` is explicitly overridden to use
    structural expression equality (`pl.Expr.meta.eq`).
    """

    input: PlanNode
    predicate: pl.Expr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Filter):
            return NotImplemented
        return self.input == other.input and self.predicate.meta.eq(other.predicate)

    def children(self) -> tuple[PlanNode, ...]:
        return (self.input,)


@dataclasses.dataclass(frozen=True, slots=True)
class Slice(PlanNode):
    """Unary plan node representing a row offset and length slice."""

    input: PlanNode
    offset: int
    length: int

    def __post_init__(self) -> None:
        for field_name, value in (("offset", self.offset), ("length", self.length)):
            if not isinstance(value, int) or isinstance(value, bool):
                msg = f"Slice {field_name} must be an int, got {type(value).__name__}."
                raise TypeError(msg)
            if value < 0:
                msg = f"Slice {field_name} must be non-negative, got {value}."
                raise ValueError(msg)

    def children(self) -> tuple[PlanNode, ...]:
        return (self.input,)


@dataclasses.dataclass(frozen=True, slots=True)
class Project(PlanNode):
    """Unary plan node representing a column projection (empty tuple = 0-col)."""

    input: PlanNode
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.columns)) != len(self.columns):
            msg = "Project columns must be unique."
            raise ValueError(msg)

    def children(self) -> tuple[PlanNode, ...]:
        return (self.input,)


LeafScan = QueryScan | TableScan
UnaryPlanOp = Filter | Slice | Project


def decompose_linear_plan(
    plan: PlanNode,
) -> tuple[LeafScan, tuple[UnaryPlanOp, ...]]:
    """Iteratively decompose a linear plan tree into `(source, ops)` bottom-to-top.

    Raises `TypeError` if an unsupported `PlanNode` subclass is encountered.
    """
    ops_reversed: list[UnaryPlanOp] = []
    curr: PlanNode = plan
    while isinstance(curr, (Filter, Slice, Project)):
        ops_reversed.append(curr)
        curr = curr.input
    if not isinstance(curr, (QueryScan, TableScan)):
        msg = f"Unsupported plan node type: {type(curr).__name__}."
        raise TypeError(msg)
    ops_reversed.reverse()
    return curr, tuple(ops_reversed)


def validate_and_normalize_linear_plan(
    plan: PlanNode,
    *,
    expected_leaf_type: type[LeafScan] | None = None,
    caller: str = "validate_and_normalize_linear_plan",
) -> tuple[LeafScan, tuple[UnaryPlanOp, ...], PlanNode]:
    """Validate uniform column visibility and collapse superseded `Project` nodes.

    Checks bottom-to-top that every `Filter` expression is expanded (no
    multi-output selectors, wildcards, or regex column projections where Polars
    `meta.root_names()` returns `[]`), every `Filter` root column and every
    `Project` column is visible in the output of the preceding node (`ValueError`
    otherwise). Superseded intermediate `Project` nodes are stripped while
    preserving the final `Project` in its original relative position, and every
    surviving operator is re-chained from `source` so
    `decompose_linear_plan(normalized_plan) == (source, normalized_ops)`.
    """
    source, ops = decompose_linear_plan(plan)
    if expected_leaf_type is not None and not isinstance(source, expected_leaf_type):
        msg = (
            f"{caller} expects a {expected_leaf_type.__name__} leaf, "
            f"got {type(source).__name__}."
        )
        raise TypeError(msg)
    visible_columns = set(source.columns)
    last_project_idx = -1

    for idx, op in enumerate(ops):
        if isinstance(op, Filter):
            if (
                op.predicate.meta.has_multiple_outputs()
                or op.predicate.meta.is_regex_projection()
            ):
                msg = (
                    "Filter predicate contains an unexpanded selector, wildcard, "
                    "or multi-column expression; expand column references explicitly "
                    "before constructing a Filter plan node."
                )
                raise ValueError(msg)
            missing = list(
                dict.fromkeys(
                    col
                    for col in op.predicate.meta.root_names()
                    if col not in visible_columns
                )
            )
            if missing:
                missing_list = ", ".join(repr(col) for col in missing)
                msg = f"Filter references unknown or hidden column(s): {missing_list}."
                raise ValueError(msg)
        elif isinstance(op, Project):
            missing = [col for col in op.columns if col not in visible_columns]
            if missing:
                missing_list = ", ".join(repr(col) for col in missing)
                msg = f"Project references unknown or hidden column(s): {missing_list}."
                raise ValueError(msg)
            visible_columns = set(op.columns)
            last_project_idx = idx

    current: PlanNode = source
    normalized_ops: list[UnaryPlanOp] = []
    for idx, op in enumerate(ops):
        if isinstance(op, Filter):
            current = Filter(input=current, predicate=op.predicate)
            normalized_ops.append(current)
        elif isinstance(op, Slice):
            current = Slice(input=current, offset=op.offset, length=op.length)
            normalized_ops.append(current)
        elif idx == last_project_idx:
            current = Project(input=current, columns=op.columns)
            normalized_ops.append(current)

    return source, tuple(normalized_ops), current


def build_io_source_plan(
    source: LeafScan,
    *,
    with_columns: Sequence[str] | None,
    predicate: pl.Expr | None,
    n_rows: int | None,
) -> PlanNode:
    """Adapt Polars `register_io_source` callback arguments into a linear plan tree."""
    node: PlanNode = source
    if n_rows is not None:
        node = Slice(input=node, offset=0, length=n_rows)
    if predicate is not None:
        node = Filter(input=node, predicate=predicate)
    output_columns = tuple(with_columns) if with_columns is not None else source.columns
    return Project(input=node, columns=output_columns)


__all__ = [
    "MAX_BIGQUERY_INT64",
    "ArrowBufferCompression",
    "Filter",
    "LeafScan",
    "PlanNode",
    "Project",
    "QueryScan",
    "Slice",
    "TableScan",
    "UnaryPlanOp",
    "build_io_source_plan",
    "decompose_linear_plan",
    "validate_and_normalize_linear_plan",
    "validate_maximum_bytes_billed",
]
