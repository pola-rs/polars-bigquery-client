"""BigQuery Storage Read API compiler and table scan executor."""

from __future__ import annotations

import dataclasses
import heapq
import logging
from collections.abc import Iterator
from typing import Any, Literal, cast

import polars as pl

from polars_bigquery.core import compiler
from polars_bigquery.core.executors.base import (
    ArrowTableClient,
    PlanExecutor,
    execute_local_plan_batches,
)
from polars_bigquery.core.plan import (
    Filter,
    PlanNode,
    Project,
    Slice,
    TableScan,
    decompose_linear_plan,
    validate_and_normalize_linear_plan,
)

logger = logging.getLogger(__name__)

MAX_ROW_RESTRICTION_BYTES = 1_000_000

_INGESTION_TIME_PSEUDO_COLUMNS_CASEFOLD = frozenset(
    {"_partitiondate", "_partitiontime"}
)

ReadApiDegradedReason = Literal[
    "row_restriction_length_pruned_conjuncts",
    "row_restriction_length_dropped_all",
]


@dataclasses.dataclass(frozen=True, slots=True)
class CompiledReadApiPlan:
    """Compiled Storage Read API `selected_fields`, `row_restriction`, and residual plan."""

    selected_fields: tuple[str, ...]
    row_restriction: str
    residual_plan: PlanNode
    degraded_reason: ReadApiDegradedReason | None = None


def _validate_max_row_restriction_bytes(max_bytes: Any) -> int:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
        msg = (
            "max_row_restriction_bytes must be a positive int, "
            f"got {type(max_bytes).__name__}."
        )
        raise TypeError(msg)
    if max_bytes <= 0:
        msg = f"max_row_restriction_bytes must be positive, got {max_bytes}."
        raise ValueError(msg)
    return max_bytes


def _dtype_fallback_rank(dtype: pl.DataType) -> int:
    """Return a cost/safety priority rank (lower is preferred) for a 0-column fallback."""
    if isinstance(dtype, pl.Boolean):
        return 0
    if isinstance(dtype, (pl.Int8, pl.UInt8)):
        return 1
    if isinstance(dtype, (pl.Int16, pl.UInt16, pl.Float16)):
        return 2
    if isinstance(dtype, (pl.Date, pl.Int32, pl.UInt32, pl.Float32)):
        return 3
    if isinstance(dtype, (pl.Int64, pl.UInt64, pl.Float64, pl.Time, pl.Datetime)):
        return 4
    if isinstance(dtype, (pl.Int128, pl.UInt128)):
        return 5
    if isinstance(dtype, pl.Decimal):
        return 5 if dtype.precision <= 38 else 8
    if isinstance(dtype, (pl.Duration, pl.Null, pl.Unknown, pl.Object)):
        return 8
    if dtype.is_nested():
        return 7
    return 6


def _select_zero_column_fallback(schema: pl.Schema) -> str:
    """Select the single narrowest safe column to request for a 0-column read."""
    physical_columns = [
        (idx, name, dtype)
        for idx, (name, dtype) in enumerate(schema.items())
        if name.casefold() not in _INGESTION_TIME_PSEUDO_COLUMNS_CASEFOLD
    ]
    if physical_columns:
        _, best_name, _ = min(
            physical_columns,
            key=lambda item: (_dtype_fallback_rank(item[2]), item[0]),
        )
        return best_name
    for name in schema:
        if name.casefold() == "_partitiontime":
            return name
    return next(iter(schema))


def _prune_conjuncts_to_budget(
    conjuncts: list[str],
    max_bytes: int,
) -> tuple[str, ReadApiDegradedReason | None]:
    """Assemble and enforce a UTF-8 byte cap on `row_restriction` conjuncts in `O(K log K)`."""
    if not conjuncts:
        return "", None

    byte_lens = [len(c.encode("utf-8")) for c in conjuncts]
    total = compiler.formatted_conjuncts_byte_length(byte_lens)
    if total <= max_bytes:
        return compiler.format_conjuncts_sql(conjuncts), None

    active = [True] * len(conjuncts)
    active_count = len(conjuncts)
    active_sum = sum(byte_lens)
    # Drop largest conjuncts first, tie-breaking rightmost-first so leftmost
    # partition/cluster filters survive.
    heap = [(-length, -idx) for idx, length in enumerate(byte_lens)]
    heapq.heapify(heap)

    while active_count > 0 and heap:
        current_bytes = (
            active_sum if active_count == 1 else active_sum + 5 * (active_count - 1) + 2
        )
        if current_bytes <= max_bytes:
            break
        neg_len, neg_idx = heapq.heappop(heap)
        idx = -neg_idx
        active[idx] = False
        active_count -= 1
        active_sum -= -neg_len

    if active_count == 0:
        return "", "row_restriction_length_dropped_all"

    kept = [c for c, keep in zip(conjuncts, active, strict=True) if keep]
    return (
        compiler.format_conjuncts_sql(kept),
        "row_restriction_length_pruned_conjuncts",
    )


def compile_read_api_plan(
    plan: PlanNode,
    *,
    max_row_restriction_bytes: int = MAX_ROW_RESTRICTION_BYTES,
) -> CompiledReadApiPlan:
    """Compile a `TableScan` logical plan into Storage Read API parameters."""
    max_bytes = _validate_max_row_restriction_bytes(max_row_restriction_bytes)
    leaf, normalized_ops, normalized_plan = validate_and_normalize_linear_plan(
        plan,
        expected_leaf_type=TableScan,
        caller="compile_read_api_plan",
    )
    source = cast(TableScan, leaf)

    final_project = next((op for op in normalized_ops if isinstance(op, Project)), None)
    output_cols = final_project.columns if final_project is not None else source.columns
    filter_cols = [
        col
        for op in normalized_ops
        if isinstance(op, Filter)
        for col in op.predicate.meta.root_names()
    ]
    selected_fields = tuple(dict.fromkeys([*output_cols, *filter_cols]))
    if not selected_fields:
        selected_fields = (_select_zero_column_fallback(source.schema),)

    conjuncts: list[str] = []
    for op in normalized_ops:
        if isinstance(op, Slice):
            if source.maintain_order or op.offset > 0:
                break
        elif isinstance(op, Filter):
            compiled_pred = compiler.compile_predicate(op.predicate)
            if compiled_pred.conjuncts:
                conjuncts.extend(compiled_pred.conjuncts)

    row_restriction, degraded_reason = _prune_conjuncts_to_budget(conjuncts, max_bytes)
    return CompiledReadApiPlan(
        selected_fields=selected_fields,
        row_restriction=row_restriction,
        residual_plan=normalized_plan,
        degraded_reason=degraded_reason,
    )


class ReadApiExecutor(PlanExecutor):
    """Executes `TableScan` logical plans via `arrow_bigquery.Client.read_table`."""

    def __init__(
        self,
        arrow_client: ArrowTableClient,
        *,
        max_row_restriction_bytes: int | None = None,
    ) -> None:
        if max_row_restriction_bytes is not None:
            max_row_restriction_bytes = _validate_max_row_restriction_bytes(
                max_row_restriction_bytes
            )
        self._arrow_client = arrow_client
        self._max_row_restriction_bytes = max_row_restriction_bytes

    def execute(
        self, plan: PlanNode, *, batch_size: int | None = None
    ) -> Iterator[pl.DataFrame]:
        active_max_bytes = (
            MAX_ROW_RESTRICTION_BYTES
            if self._max_row_restriction_bytes is None
            else self._max_row_restriction_bytes
        )
        compiled = compile_read_api_plan(
            plan, max_row_restriction_bytes=active_max_bytes
        )
        source_leaf, residual_ops = decompose_linear_plan(compiled.residual_plan)
        source = cast(TableScan, source_leaf)

        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        if compiled.degraded_reason is not None or debug_enabled:
            restriction_bytes = len(compiled.row_restriction.encode("utf-8"))
            if compiled.degraded_reason is not None:
                logger.warning(
                    "Read API row_restriction pushdown degraded (%s): "
                    "restriction_bytes=%d, max_row_restriction_bytes=%d",
                    compiled.degraded_reason,
                    restriction_bytes,
                    active_max_bytes,
                )
            if debug_enabled:
                logger.debug(
                    "Executing scan_table: maintain_order=%s, selected_fields=%d/%d, "
                    "row_restriction_bytes=%d, degraded_reason=%s",
                    source.maintain_order,
                    len(compiled.selected_fields),
                    len(source.columns),
                    restriction_bytes,
                    compiled.degraded_reason,
                )

        arrow_stream_exporter = self._arrow_client.read_table(
            source.table_ref,
            maintain_order=source.maintain_order,
            selected_fields=list(compiled.selected_fields),
            row_restriction=compiled.row_restriction,
        )
        return execute_local_plan_batches(
            pl.scan_arrow_c_stream(arrow_stream_exporter),
            residual_ops,
            batch_size=batch_size,
            maintain_order=source.maintain_order,
        )


__all__ = [
    "MAX_ROW_RESTRICTION_BYTES",
    "CompiledReadApiPlan",
    "ReadApiDegradedReason",
    "ReadApiExecutor",
    "compile_read_api_plan",
]
