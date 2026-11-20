"""GoogleSQL pipe compiler and query scan executor."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator, Sequence
from typing import Any, Literal, cast

import polars as pl

from polars_bigquery.core import compiler
from polars_bigquery.core.executors.base import PlanExecutor, apply_local_plan
from polars_bigquery.core.plan import (
    Filter,
    PlanNode,
    Project,
    QueryScan,
    Slice,
    UnaryPlanOp,
    decompose_linear_plan,
    validate_and_normalize_linear_plan,
)

logger = logging.getLogger(__name__)

MAX_UNRESOLVED_SQL_BYTES = 1_000_000
MAX_BIGQUERY_LIMIT = (1 << 63) - 1

DegradedReason = Literal[
    "sql_length_pruned_conjuncts",
    "sql_length_dropped_where",
    "sql_length_skipped_pushdown",
]


@dataclasses.dataclass(frozen=True, slots=True)
class CompiledQueryPlan:
    """Compiled GoogleSQL pipe query, pushdown metadata, and residual local plan."""

    sql: str
    rewritten: bool
    residual_plan: PlanNode
    projected_columns: tuple[str, ...] | None
    where_clauses: tuple[str, ...]
    pushed_slices: tuple[tuple[int, int], ...]
    dummy_column: str | None
    degraded_reason: DegradedReason | None


def _choose_dummy_column_name(all_columns: Sequence[str]) -> str:
    """Pick a collision-safe case-insensitive alias for zero-column projections."""
    existing = {col.casefold() for col in all_columns}
    base = "_polars_bigquery_empty"
    if base not in existing:
        return base
    suffix = 1
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"


def _lower_and_assemble(
    *,
    source: QueryScan,
    normalized_ops: tuple[UnaryPlanOp, ...],
    normalized_plan: PlanNode,
    inner_query: str,
    select_clause: str | None,
    projected_columns: tuple[str, ...] | None,
    dummy_column: str | None,
    filter_states: dict[int, tuple[tuple[str, ...], bool]],
    degraded_reason: DegradedReason | None = None,
) -> tuple[CompiledQueryPlan, frozenset[int]]:
    """Lower `Filter`/`Slice` stages into GoogleSQL pipe operators and residual plan."""
    pipe_clauses: list[str] = []
    where_clauses: list[str] = []
    pushed_slices: list[tuple[int, int]] = []
    emitted_filter_indices: set[int] = set()
    last_where_idx: int | None = None

    pending_zero_offset_slice: int | None = None
    partial_filter_barrier = False
    unpushed_slice_seen = False

    residual_plan: PlanNode = source

    def flush_pending_slice() -> None:
        nonlocal pending_zero_offset_slice
        if pending_zero_offset_slice is not None:
            pipe_clauses.append(f"|> LIMIT {pending_zero_offset_slice}")
            pushed_slices.append((0, pending_zero_offset_slice))
            pending_zero_offset_slice = None

    for op_idx, op in enumerate(normalized_ops):
        if isinstance(op, Project):
            residual_plan = Project(input=residual_plan, columns=op.columns)
            continue

        if unpushed_slice_seen:
            if isinstance(op, Filter):
                residual_plan = Filter(input=residual_plan, predicate=op.predicate)
            else:
                residual_plan = Slice(
                    input=residual_plan, offset=op.offset, length=op.length
                )
            continue

        if isinstance(op, Filter):
            residual_plan = Filter(input=residual_plan, predicate=op.predicate)
            conjuncts, is_exact = filter_states[op_idx]
            clause = compiler.format_conjuncts_sql(conjuncts)
            if clause:
                last_where_idx = len(pipe_clauses)
                pipe_clauses.append(f"|> WHERE {clause}")
                where_clauses.append(clause)
                emitted_filter_indices.add(op_idx)
            if not is_exact:
                flush_pending_slice()
                partial_filter_barrier = True
            continue

        # op is Slice
        flush_pending_slice()
        if (
            partial_filter_barrier
            or op.offset > MAX_BIGQUERY_LIMIT
            or op.length > MAX_BIGQUERY_LIMIT
        ):
            unpushed_slice_seen = True
            residual_plan = Slice(
                input=residual_plan, offset=op.offset, length=op.length
            )
        elif op.offset == 0:
            pending_zero_offset_slice = op.length
            residual_plan = Slice(input=residual_plan, offset=0, length=op.length)
        else:
            pipe_clauses.append(f"|> LIMIT {op.length} OFFSET {op.offset}")
            pushed_slices.append((op.offset, op.length))
            residual_plan = Slice(input=residual_plan, offset=0, length=op.length)

    flush_pending_slice()

    if select_clause is not None:
        insert_at = 0 if last_where_idx is None else last_where_idx + 1
        pipe_clauses.insert(insert_at, select_clause)

    if not pipe_clauses:
        return (
            CompiledQueryPlan(
                sql=source.raw_query,
                rewritten=False,
                residual_plan=normalized_plan,
                projected_columns=None,
                where_clauses=(),
                pushed_slices=(),
                dummy_column=None,
                degraded_reason=degraded_reason,
            ),
            frozenset(),
        )

    assembled_sql = "\n".join([inner_query, *pipe_clauses])
    return (
        CompiledQueryPlan(
            sql=assembled_sql,
            rewritten=True,
            residual_plan=residual_plan,
            projected_columns=projected_columns,
            where_clauses=tuple(where_clauses),
            pushed_slices=tuple(pushed_slices),
            dummy_column=dummy_column,
            degraded_reason=degraded_reason,
        ),
        frozenset(emitted_filter_indices),
    )


def compile_query_plan(
    plan: PlanNode, *, max_sql_bytes: int = MAX_UNRESOLVED_SQL_BYTES
) -> CompiledQueryPlan:
    """Compile a `QueryScan` logical plan into a GoogleSQL pipe query."""
    leaf, _ = decompose_linear_plan(plan)
    if not isinstance(leaf, QueryScan):
        msg = f"compile_query_plan expects a QueryScan leaf, got {type(leaf).__name__}."
        raise TypeError(msg)

    _, normalized_ops, normalized_plan = validate_and_normalize_linear_plan(plan)
    source = leaf

    if source.maintain_order:
        return CompiledQueryPlan(
            sql=source.raw_query,
            rewritten=False,
            residual_plan=normalized_plan,
            projected_columns=None,
            where_clauses=(),
            pushed_slices=(),
            dummy_column=None,
            degraded_reason=None,
        )

    inner_query = source.raw_query.rstrip().removesuffix(";").rstrip()

    final_project = next((op for op in normalized_ops if isinstance(op, Project)), None)
    output_cols = final_project.columns if final_project is not None else source.columns
    needed_set = set(output_cols) | {
        col
        for op in normalized_ops
        if isinstance(op, Filter)
        for col in op.predicate.meta.root_names()
    }
    needed_columns = tuple(col for col in source.columns if col in needed_set)

    select_clause: str | None
    projected_columns: tuple[str, ...] | None
    dummy_column: str | None = None
    if not needed_columns:
        dummy_column = _choose_dummy_column_name(source.columns)
        dummy_ident = compiler.sql.column_to_sql_identifier(dummy_column)
        select_clause = f"|> SELECT 1 AS {dummy_ident}"
        projected_columns = ()
    elif not source.explicit_schema and len(needed_columns) == len(source.columns):
        select_clause = None
        projected_columns = None
    else:
        projected_sql = ", ".join(
            compiler.sql.column_to_sql_identifier(col) for col in needed_columns
        )
        select_clause = f"|> SELECT {projected_sql}"
        projected_columns = needed_columns

    initial_filter_states: dict[int, tuple[tuple[str, ...], bool]] = {}
    for op_idx, op in enumerate(normalized_ops):
        if isinstance(op, Filter):
            compiled_pred = compiler.compile_predicate(op.predicate)
            initial_filter_states[op_idx] = (
                compiled_pred.conjuncts,
                compiled_pred.is_exact,
            )

    candidate, emitted_filter_indices = _lower_and_assemble(
        source=source,
        normalized_ops=normalized_ops,
        normalized_plan=normalized_plan,
        inner_query=inner_query,
        select_clause=select_clause,
        projected_columns=projected_columns,
        dummy_column=dummy_column,
        filter_states=initial_filter_states,
    )

    if not candidate.rewritten:
        return candidate

    # Tier 1: Full piped SQL fits within UTF-8 byte ceiling.
    if len(candidate.sql.encode("utf-8")) <= max_sql_bytes:
        return candidate

    if candidate.where_clauses:
        # Tier 2: Prune the largest emitted top-level AND conjuncts first.
        tier2_states = dict(initial_filter_states)
        while (
            len(candidate.sql.encode("utf-8")) > max_sql_bytes
            and candidate.where_clauses
        ):
            best_op_idx, best_conj_idx = max(
                (
                    (op_idx, conj_idx)
                    for op_idx in emitted_filter_indices
                    for conj_idx in range(len(tier2_states[op_idx][0]))
                ),
                key=lambda item: (
                    len(tier2_states[item[0]][0][item[1]].encode("utf-8")),
                    item[0],
                    item[1],
                ),
            )
            old_conjuncts, _ = tier2_states[best_op_idx]
            tier2_states[best_op_idx] = (
                old_conjuncts[:best_conj_idx] + old_conjuncts[best_conj_idx + 1 :],
                False,
            )
            candidate, emitted_filter_indices = _lower_and_assemble(
                source=source,
                normalized_ops=normalized_ops,
                normalized_plan=normalized_plan,
                inner_query=inner_query,
                select_clause=select_clause,
                projected_columns=projected_columns,
                dummy_column=dummy_column,
                filter_states=tier2_states,
            )

        if candidate.where_clauses:
            return dataclasses.replace(
                candidate, degraded_reason="sql_length_pruned_conjuncts"
            )

        # Tier 3: Drop all remaining |> WHERE clauses while preserving trivial filters.
        tier3_states = {
            op_idx: ((), False) if state[0] else state
            for op_idx, state in initial_filter_states.items()
        }
        tier3_candidate, _ = _lower_and_assemble(
            source=source,
            normalized_ops=normalized_ops,
            normalized_plan=normalized_plan,
            inner_query=inner_query,
            select_clause=select_clause,
            projected_columns=projected_columns,
            dummy_column=dummy_column,
            filter_states=tier3_states,
            degraded_reason="sql_length_dropped_where",
        )
        if (
            tier3_candidate.rewritten
            and len(tier3_candidate.sql.encode("utf-8")) <= max_sql_bytes
        ):
            return tier3_candidate

    # Tier 4: Verbatim fallback to raw_query with normalized full residual plan.
    return CompiledQueryPlan(
        sql=source.raw_query,
        rewritten=False,
        residual_plan=normalized_plan,
        projected_columns=None,
        where_clauses=(),
        pushed_slices=(),
        dummy_column=None,
        degraded_reason="sql_length_skipped_pushdown",
    )


class QueryExecutor(PlanExecutor):
    """Executes `QueryScan` logical plans via `arrow_bigquery.Client.read_query`."""

    def __init__(self, arrow_client: Any, *, max_sql_bytes: int | None = None) -> None:
        self._arrow_client = arrow_client
        self._max_sql_bytes = max_sql_bytes

    def execute(
        self, plan: PlanNode, *, batch_size: int | None = None
    ) -> Iterator[pl.DataFrame]:
        active_max_sql_bytes = (
            MAX_UNRESOLVED_SQL_BYTES
            if self._max_sql_bytes is None
            else self._max_sql_bytes
        )
        compiled = compile_query_plan(plan, max_sql_bytes=active_max_sql_bytes)
        source_leaf, _ = decompose_linear_plan(compiled.residual_plan)
        source = cast(QueryScan, source_leaf)

        total_columns = len(source.columns)
        if compiled.projected_columns is None:
            projection_summary = f"{total_columns}/{total_columns}"
        elif len(compiled.projected_columns) == 0:
            projection_summary = "0-col dummy"
        else:
            projection_summary = f"{len(compiled.projected_columns)}/{total_columns}"

        sql_bytes = len(compiled.sql.encode("utf-8"))
        where_bytes = sum(len(c.encode("utf-8")) for c in compiled.where_clauses)

        if compiled.degraded_reason is not None:
            logger.warning(
                "Query pushdown degraded (%s): sql_bytes=%d, max_sql_bytes=%d",
                compiled.degraded_reason,
                sql_bytes,
                active_max_sql_bytes,
            )

        logger.debug(
            "Executing scan_query: maintain_order=%s, rewritten=%s, projected=%s, "
            "where_clauses=%d, where_bytes=%d, pushed_slices=%s, sql_bytes=%d, "
            "degraded_reason=%s",
            source.maintain_order,
            compiled.rewritten,
            projection_summary,
            len(compiled.where_clauses),
            where_bytes,
            compiled.pushed_slices,
            sql_bytes,
            compiled.degraded_reason,
        )

        arrow_stream_exporter = self._arrow_client.read_query(
            compiled.sql,
            arrow_buffer_compression=source.arrow_buffer_compression,
            dry_run=False,
            maintain_order=source.maintain_order,
            use_query_cache=source.use_query_cache,
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
    "MAX_BIGQUERY_LIMIT",
    "MAX_UNRESOLVED_SQL_BYTES",
    "CompiledQueryPlan",
    "DegradedReason",
    "QueryExecutor",
    "compile_query_plan",
]
