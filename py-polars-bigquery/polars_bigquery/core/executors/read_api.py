"""BigQuery Storage Read API compiler and table scan executor."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any, cast

import polars as pl

from polars_bigquery.core import compiler
from polars_bigquery.core.executors.base import PlanExecutor, apply_local_plan
from polars_bigquery.core.plan import (
    Filter,
    PlanNode,
    Project,
    Slice,
    TableScan,
    decompose_linear_plan,
    validate_and_normalize_linear_plan,
)

_INGESTION_TIME_PSEUDO_COLUMNS = frozenset({"_PARTITIONDATE", "_PARTITIONTIME"})


@dataclasses.dataclass(frozen=True, slots=True)
class CompiledReadApiPlan:
    """Compiled Storage Read API `selected_fields`, `row_restriction`, and residual plan."""

    selected_fields: tuple[str, ...]
    row_restriction: str
    residual_plan: PlanNode


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
        if name not in _INGESTION_TIME_PSEUDO_COLUMNS
    ]
    if physical_columns:
        _, best_name, _ = min(
            physical_columns,
            key=lambda item: (_dtype_fallback_rank(item[2]), item[0]),
        )
        return best_name
    if "_PARTITIONTIME" in schema:
        return "_PARTITIONTIME"
    return "_PARTITIONDATE"


def compile_read_api_plan(plan: PlanNode) -> CompiledReadApiPlan:
    """Compile a `TableScan` logical plan into Storage Read API parameters."""
    leaf, _ = decompose_linear_plan(plan)
    if not isinstance(leaf, TableScan):
        msg = (
            "compile_read_api_plan expects a TableScan leaf, "
            f"got {type(leaf).__name__}."
        )
        raise TypeError(msg)

    _, normalized_ops, normalized_plan = validate_and_normalize_linear_plan(plan)
    source = leaf

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

    clauses: list[str] = []
    for op in normalized_ops:
        if isinstance(op, Slice):
            if source.maintain_order or op.offset > 0:
                break
        elif isinstance(op, Filter):
            clause = compiler.compile_predicate(op.predicate).sql
            if clause:
                clauses.append(clause)

    row_restriction = compiler.format_conjuncts_sql(clauses)
    return CompiledReadApiPlan(
        selected_fields=selected_fields,
        row_restriction=row_restriction,
        residual_plan=normalized_plan,
    )


class ReadApiExecutor(PlanExecutor):
    """Executes `TableScan` logical plans via `arrow_bigquery.Client.read_table`."""

    def __init__(self, arrow_client: Any) -> None:
        self._arrow_client = arrow_client

    def execute(
        self, plan: PlanNode, *, batch_size: int | None = None
    ) -> Iterator[pl.DataFrame]:
        compiled = compile_read_api_plan(plan)
        source_leaf, _ = decompose_linear_plan(compiled.residual_plan)
        source = cast(TableScan, source_leaf)

        arrow_stream_exporter = self._arrow_client.read_table(
            source.table_ref,
            maintain_order=source.maintain_order,
            selected_fields=list(compiled.selected_fields),
            row_restriction=compiled.row_restriction,
        )
        lazyframe = apply_local_plan(
            pl.scan_arrow_c_stream(arrow_stream_exporter),
            compiled.residual_plan,
        )
        return lazyframe.collect_batches(
            chunk_size=batch_size,
            maintain_order=source.maintain_order,
        )


__all__ = [
    "CompiledReadApiPlan",
    "ReadApiExecutor",
    "compile_read_api_plan",
]
