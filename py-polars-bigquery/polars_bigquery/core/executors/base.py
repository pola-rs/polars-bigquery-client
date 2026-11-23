"""Shared backend executor interface, client protocols, and local Polars residual evaluator."""

from __future__ import annotations

import abc
import functools
from collections.abc import Iterator
from typing import Any, Protocol

import arrow_bigquery
import polars as pl

from polars_bigquery.core.plan import (
    Filter,
    PlanNode,
    Project,
    Slice,
    UnaryPlanOp,
    decompose_linear_plan,
)

# Polars native `.slice(offset, length)` converts `offset` via signed 64-bit
# integer (`i64`) and `length` via the runtime's `IdxSize` (`u32` on default
# `polars-runtime-32`, `u64`/`i64` on `polars-runtime-64`).
_POLARS_MAX_SLICE_OFFSET = (1 << 63) - 1


@functools.cache
def _polars_max_slice_length() -> int:
    """Lazily detect the maximum `length` accepted by `pl.LazyFrame.slice` in this runtime."""
    probe = pl.LazyFrame({"_p": [1]})
    try:
        probe.slice(0, 1 << 32)
    except OverflowError:
        return (1 << 32) - 1
    return (1 << 63) - 1


class ArrowQueryClient(Protocol):
    """Minimal protocol satisfied by `arrow_bigquery.Client` for query execution."""

    def read_query(
        self,
        query: str,
        *,
        arrow_buffer_compression: str = ...,
        dry_run: bool = ...,
        maintain_order: bool = ...,
        use_query_cache: bool | None = ...,
        maximum_bytes_billed: int | None = ...,
    ) -> Any:
        """Execute a GoogleSQL query (or dry run) and return an Arrow C Stream exporter."""


def close_exporter(exporter: Any) -> None:
    """Close an `ArrowStreamExporter` immediately when supported to abort background streams."""
    close = getattr(exporter, "close", None)
    if callable(close):
        close()


class ArrowTableClient(Protocol):
    """Minimal protocol satisfied by `arrow_bigquery.Client` for table scans."""

    def read_table(
        self,
        table: arrow_bigquery.BigQueryTableId,
        *,
        maintain_order: bool = ...,
        max_stream_count: int | None = ...,
        selected_fields: list[str] | None = ...,
        row_restriction: str = ...,
    ) -> Any:
        """Open a BigQuery Storage Read API stream and return an Arrow C Stream exporter."""


class PlanExecutor(abc.ABC):
    """Abstract base class for BigQuery logical scan plan executors."""

    @abc.abstractmethod
    def execute(
        self, plan: PlanNode, *, batch_size: int | None = None
    ) -> Iterator[pl.DataFrame]:
        """Execute `plan` and yield streamed Polars `DataFrame` batches."""


def _slice_fits_polars(op: Slice) -> bool:
    """Return `True` if `op` fits within native `pl.LazyFrame.slice` integer bounds."""
    return op.offset <= _POLARS_MAX_SLICE_OFFSET and (
        op.length <= (1 << 32) - 1 or op.length <= _polars_max_slice_length()
    )


def _apply_ops_to_frame(
    lazyframe: pl.LazyFrame, ops: tuple[UnaryPlanOp, ...]
) -> pl.LazyFrame:
    """Apply `ops` to `lazyframe` after verifying all `Slice` bounds fit native Polars."""
    result = lazyframe
    for op in ops:
        if isinstance(op, Slice):
            if not _slice_fits_polars(op):
                max_length = _polars_max_slice_length()
                msg = (
                    f"Slice(offset={op.offset}, length={op.length}) exceeds "
                    f"Polars native slice limits "
                    f"(max offset={_POLARS_MAX_SLICE_OFFSET}, "
                    f"max length={max_length})."
                )
                raise ValueError(msg)
            result = result.slice(op.offset, op.length)
        elif isinstance(op, Filter):
            result = result.filter(op.predicate)
        else:
            result = (
                result.select(list(op.columns)) if op.columns else result.drop(pl.all())
            )
    return result


def apply_local_plan(lazyframe: pl.LazyFrame, plan: PlanNode) -> pl.LazyFrame:
    """Apply the normalized residual operators of `plan` bottom-to-top."""
    _, ops = decompose_linear_plan(plan)
    return _apply_ops_to_frame(lazyframe, ops)


def _close_iterator(iterator: Iterator[pl.DataFrame]) -> None:
    """Close an upstream generator/iterator eagerly if it exposes `.close()`."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _iter_slice_batches(
    batches: Iterator[pl.DataFrame],
    offset: int,
    length: int,
) -> Iterator[pl.DataFrame]:
    """Lazily apply an arbitrary-precision `Slice(offset, length)` to a batch iterator."""
    try:
        if length == 0:
            return

        skipped = 0
        emitted = 0
        for batch in batches:
            height = batch.height
            if height == 0:
                continue
            start = 0
            if skipped < offset:
                drop_rows = min(height, offset - skipped)
                skipped += drop_rows
                start = drop_rows
                if start >= height:
                    continue
            remaining = length - emitted
            take = min(height - start, remaining)
            if take > 0:
                yield (
                    batch if start == 0 and take == height else batch.slice(start, take)
                )
                emitted += take
            if emitted >= length:
                break
    finally:
        _close_iterator(batches)


def _apply_project_batch(batch: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    return batch.select(list(columns)) if columns else batch.drop(batch.columns)


def _iter_filter_batches(
    batches: Iterator[pl.DataFrame],
    predicate: pl.Expr,
) -> Iterator[pl.DataFrame]:
    """Lazily filter batches, binding `predicate` per call frame (no loop-closure aliasing)."""
    try:
        for raw in batches:
            batch = raw.filter(predicate)
            if batch.height > 0:
                yield batch
    finally:
        _close_iterator(batches)


def _iter_project_batches(
    batches: Iterator[pl.DataFrame],
    columns: tuple[str, ...],
) -> Iterator[pl.DataFrame]:
    """Lazily project batches, binding `columns` per call frame (no loop-closure aliasing)."""
    try:
        for raw in batches:
            yield _apply_project_batch(raw, columns)
    finally:
        _close_iterator(batches)


def _iter_apply_ops(
    batches: Iterator[pl.DataFrame],
    ops: tuple[UnaryPlanOp, ...],
) -> Iterator[pl.DataFrame]:
    """Stream arbitrary-order residual `Slice`/`Filter`/`Project` over a batch iterator."""
    stream = batches
    for op in ops:
        if isinstance(op, Slice):
            stream = _iter_slice_batches(stream, op.offset, op.length)
        elif isinstance(op, Filter):
            stream = _iter_filter_batches(stream, op.predicate)
        else:
            stream = _iter_project_batches(stream, op.columns)
    return stream


def execute_local_plan_batches(
    lazyframe: pl.LazyFrame,
    ops: tuple[UnaryPlanOp, ...],
    *,
    batch_size: int | None = None,
    maintain_order: bool = False,
) -> Iterator[pl.DataFrame]:
    """Execute residual `ops` over `lazyframe`, safely handling `> u32::MAX` / `> i64::MAX` slices.

    When all `Slice` operators fit inside native Polars `slice(offset, length)` bounds,
    delegates directly to `lazyframe.slice / filter / select` followed by
    `collect_batches(chunk_size=batch_size, maintain_order=maintain_order)` using default
    Polars optimizer flags. When any `Slice` exceeds Polars' native integer bounds
    (for example a `u64::MAX` sentinel or `length >= 2**32` on `polars-runtime-32`),
    applies the leading native prefix inside Polars and evaluates the oversized
    `Slice` stage and subsequent operators incrementally with arbitrary-precision
    Python integers so `OverflowError` cannot occur.
    """
    # Fall back to incremental Python-side batch streaming at the first
    # operator that hits a Polars runtime limitation: either a `Slice` whose
    # bounds exceed native `i64`/`IdxSize`, or an empty `Project(())` where
    # Polars 2.0–2.1 `lazyframe.drop(pl.all()).collect_batches()` loses row height.
    split_at = next(
        (
            idx
            for idx, op in enumerate(ops)
            if (isinstance(op, Slice) and not _slice_fits_polars(op))
            or (isinstance(op, Project) and len(op.columns) == 0)
        ),
        None,
    )
    if split_at is None:
        return _apply_ops_to_frame(lazyframe, ops).collect_batches(
            chunk_size=batch_size,
            maintain_order=maintain_order,
        )

    prefix_lf = _apply_ops_to_frame(lazyframe, ops[:split_at])
    base_iter = prefix_lf.collect_batches(
        chunk_size=batch_size,
        maintain_order=maintain_order,
    )
    return _iter_apply_ops(base_iter, ops[split_at:])


__all__ = [
    "ArrowQueryClient",
    "ArrowTableClient",
    "PlanExecutor",
    "apply_local_plan",
    "close_exporter",
    "execute_local_plan_batches",
]
