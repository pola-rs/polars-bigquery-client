"""Shared backend executor interface and local Polars residual evaluator."""

from __future__ import annotations

import abc
from collections.abc import Iterator

import polars as pl

from polars_bigquery.core.plan import (
    Filter,
    PlanNode,
    Slice,
    decompose_linear_plan,
)


class PlanExecutor(abc.ABC):
    """Abstract base class for BigQuery logical scan plan executors."""

    @abc.abstractmethod
    def execute(
        self, plan: PlanNode, *, batch_size: int | None = None
    ) -> Iterator[pl.DataFrame]:
        """Execute `plan` and yield streamed Polars `DataFrame` batches."""


def apply_local_plan(lazyframe: pl.LazyFrame, plan: PlanNode) -> pl.LazyFrame:
    """Apply the normalized residual operators of `plan` bottom-to-top."""
    _, ops = decompose_linear_plan(plan)
    result = lazyframe
    for op in ops:
        if isinstance(op, Slice):
            result = result.slice(op.offset, op.length)
        elif isinstance(op, Filter):
            result = result.filter(op.predicate)
        else:
            result = (
                result.select(list(op.columns)) if op.columns else result.drop(pl.all())
            )
    return result


__all__ = [
    "PlanExecutor",
    "apply_local_plan",
]
