"""GoogleSQL pipe compiler and query scan executor."""

from __future__ import annotations

import dataclasses
import heapq
import logging
import re
from collections.abc import Iterator, Sequence
from typing import Any, Literal, cast

import polars as pl

from polars_bigquery.core import compiler
from polars_bigquery.core.executors.base import (
    ArrowQueryClient,
    PlanExecutor,
    execute_local_plan_batches,
)
from polars_bigquery.core.plan import (
    MAX_BIGQUERY_INT64,
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
MAX_BIGQUERY_LIMIT = MAX_BIGQUERY_INT64

_WHERE_LINE_PREFIX_BYTES = len(b"\n|> WHERE ")

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


_SQL_WS_RE = re.compile(r"\s+")
_SQL_SPECIAL_RE = re.compile(r"""--|[;#`'"]|/\*""")


def _scan_line_comment_end(sql: str, start: int) -> int:
    """Return the index immediately past the first `\\r` or `\\n` ending a line comment."""
    nl = sql.find("\n", start)
    cr = sql.find("\r", start)
    if nl == -1:
        return len(sql) if cr == -1 else cr + 1
    if cr == -1 or nl < cr:
        return nl + 1
    return cr + 1


def _scan_quoted_literal(sql: str, start: int, delim: str) -> int | None:
    """Advance past a quoted string literal or backtick identifier in strict `O(N)`.

    Caches `str.find` positions so every character in `sql` is visited at most
    once, even on adversarial runs of backslashes. Backslashes escape the next
    character for both standard and raw GoogleSQL string literals (`r'...'`,
    `r\"...\"`, triple-quoted variants), matching BigQuery/GoogleSQL's lexical
    rule that a literal cannot terminate right after an odd number of backslashes.

    Returns the index immediately past the closing delimiter, or `None` if the
    literal reaches EOF unterminated.
    """
    dlen = len(delim)
    next_bs = sql.find("\\", start)
    next_delim = sql.find(delim, start)

    while next_delim != -1:
        if next_bs != -1 and next_bs < next_delim:
            # Since `next_bs < next_delim <= len(sql) - dlen`, `next_bs + 2 <= len(sql)`.
            i = next_bs + 2
            next_bs = sql.find("\\", i)
            if next_delim < i:
                next_delim = sql.find(delim, i)
        else:
            return next_delim + dlen

    return None


def _scan_sql_trailing_semicolon(sql: str) -> tuple[bool, int | None]:
    """Scan GoogleSQL tokens in strict `O(len(sql))` using C-accelerated jumps.

    Returns ``(has_non_semicolon_token, last_trailing_semicolon_index)``.
    `last_trailing_semicolon_index` is the index of a single `;` token whose
    only following content consists of whitespace and well-formed comments
    (`-- ...`, `# ...`, or closed `/* ... */`), or `None` if the last
    significant token is not `;` or an unterminated block comment/string/identifier
    reaches EOF (so syntax errors are never silently discarded).
    """
    n = len(sql)
    i = 0
    has_non_semi_token = False
    last_trailing_semi_idx: int | None = None

    while i < n:
        # Whitespace trivia
        ws_match = _SQL_WS_RE.match(sql, i)
        if ws_match is not None:
            i = ws_match.end()
            continue

        ch = sql[i]

        # Line comment: `-- ...` (terminated by `\n` or `\r`)
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            i = _scan_line_comment_end(sql, i + 2)
            continue

        # Line comment: `# ...` (terminated by `\n` or `\r`)
        if ch == "#":
            i = _scan_line_comment_end(sql, i + 1)
            continue

        # Block comment: `/* ... */` (non-nesting in GoogleSQL)
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            if end == -1:
                # Unterminated block comment: never strip a preceding semicolon,
                # so BigQuery still surfaces the syntax error.
                return has_non_semi_token, None
            i = end + 2
            continue

        # Semicolon statement terminator
        if ch == ";":
            last_trailing_semi_idx = i
            i += 1
            continue

        # Any other token marks non-empty SQL and clears a previous trailing `;`
        has_non_semi_token = True
        last_trailing_semi_idx = None

        # Backtick-quoted identifier
        if ch == "`":
            after = _scan_quoted_literal(sql, i + 1, "`")
            if after is None:
                return True, None
            i = after
            continue

        # Single / double / triple-quoted string literals (including r/b/rb/br
        # prefixed raw and byte literals, which share backslash quote-escaping).
        if ch in ("'", '"'):
            quote = ch
            if i + 2 < n and sql[i + 1] == quote and sql[i + 2] == quote:
                after = _scan_quoted_literal(sql, i + 3, quote * 3)
            else:
                after = _scan_quoted_literal(sql, i + 1, quote)
            if after is None:
                return True, None
            i = after
            continue

        # Jump in C past ordinary SQL tokens directly to the next comment,
        # quote delimiter, backtick, or semicolon.
        match = _SQL_SPECIAL_RE.search(sql, i + 1)
        i = n if match is None else match.start()

    return has_non_semi_token, last_trailing_semi_idx


def has_non_comment_sql(sql: str) -> bool:
    """Return `True` if `sql` contains at least one token besides whitespace, comments, and `;`."""
    has_token, _ = _scan_sql_trailing_semicolon(sql)
    return has_token


def _strip_single_trailing_semicolon(sql: str) -> str:
    """Right-trim `sql` and remove at most one trailing bare statement-terminating `;`.

    Handles comments placed *after* the trailing semicolon (e.g. `SELECT 1; -- note`,
    `SELECT 1; # note`, `SELECT 1; /* trace_id: 123 */`) **without discarding those
    comments**, so tracing/audit metadata (such as SQLCommenter/dbt tags) is kept
    intact. Unterminated block comments (`SELECT 1; /* unterminated`) are left
    untouched so BigQuery rejects the syntax error rather than executing them.
    Leading whitespace/newlines stay intact so `[line:col]` error coordinates remain
    unshifted, `;` inside strings/comments/identifiers are untouched, and preceding `;`
    from invalid double-semicolon SQL (`SELECT 1;;`) remain so BigQuery rejects them.
    """
    _, semi_idx = _scan_sql_trailing_semicolon(sql)
    if semi_idx is not None:
        return f"{sql[:semi_idx]}{sql[semi_idx + 1 :]}".rstrip()
    return sql.rstrip()


def _validate_max_sql_bytes(max_sql_bytes: Any) -> int:
    if not isinstance(max_sql_bytes, int) or isinstance(max_sql_bytes, bool):
        msg = (
            f"max_sql_bytes must be a positive int, got {type(max_sql_bytes).__name__}."
        )
        raise TypeError(msg)
    if max_sql_bytes <= 0:
        msg = f"max_sql_bytes must be positive, got {max_sql_bytes}."
        raise ValueError(msg)
    return max_sql_bytes


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


def _where_clause_line_bytes(conjunct_sum: int, conjunct_count: int) -> int:
    """Return exact UTF-8 bytes contributed by `\\n|> WHERE <formatted_conjuncts>`."""
    if conjunct_count == 0:
        return 0
    return (
        _WHERE_LINE_PREFIX_BYTES
        + compiler.formatted_conjuncts_byte_length_from_stats(
            conjunct_count, conjunct_sum
        )
    )


def _compute_active_lower_summary(
    normalized_ops: tuple[UnaryPlanOp, ...],
    filter_counts: dict[int, tuple[int, int, bool]],
    inner_bytes: int,
    select_line_bytes: int,
) -> tuple[int, set[int]]:
    """Compute exact assembled UTF-8 byte length and emitted filter indices without string concat.

    `filter_counts[op_idx]` holds ``(active_count, active_byte_sum, is_exact)``.
    Mirrors `_lower_and_assemble`'s exact state machine in `O(len(normalized_ops))`.
    """
    total_bytes = inner_bytes + select_line_bytes
    emitted_filter_indices: set[int] = set()
    has_clause = select_line_bytes > 0

    pending_zero_offset_slice: int | None = None
    partial_filter_barrier = False

    def flush_slice() -> None:
        nonlocal pending_zero_offset_slice, total_bytes, has_clause
        if pending_zero_offset_slice is not None:
            # "\n|> LIMIT <n>"
            total_bytes += 10 + len(str(pending_zero_offset_slice))
            has_clause = True
            pending_zero_offset_slice = None

    for op_idx, op in enumerate(normalized_ops):
        if isinstance(op, Project):
            continue
        if isinstance(op, Filter):
            count, byte_sum, is_exact = filter_counts[op_idx]
            if count > 0:
                total_bytes += _where_clause_line_bytes(byte_sum, count)
                emitted_filter_indices.add(op_idx)
                has_clause = True
            if not is_exact:
                flush_slice()
                partial_filter_barrier = True
            continue
        # op is Slice
        flush_slice()
        if (
            partial_filter_barrier
            or op.offset > MAX_BIGQUERY_LIMIT
            or op.length > MAX_BIGQUERY_LIMIT
        ):
            break
        if op.offset == 0:
            pending_zero_offset_slice = op.length
        else:
            # "\n|> LIMIT <length> OFFSET <offset>"
            total_bytes += 18 + len(str(op.length)) + len(str(op.offset))
            has_clause = True

    flush_slice()
    return (total_bytes if has_clause else inner_bytes), emitted_filter_indices


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
    max_sql_bytes = _validate_max_sql_bytes(max_sql_bytes)

    leaf, normalized_ops, normalized_plan = validate_and_normalize_linear_plan(
        plan,
        expected_leaf_type=QueryScan,
        caller="compile_query_plan",
    )
    source = cast(QueryScan, leaf)

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

    inner_query = _strip_single_trailing_semicolon(source.raw_query)

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
    candidate_bytes = len(candidate.sql.encode("utf-8"))
    if candidate_bytes <= max_sql_bytes:
        return candidate

    # Fast-path: if `inner_query` alone already hits or exceeds the ceiling,
    # no rewritten `inner_query\n|> ...` variant can fit in Tier 2 or Tier 3.
    inner_bytes = len(inner_query.encode("utf-8"))
    if inner_bytes >= max_sql_bytes:
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

    if candidate.where_clauses:
        # Tier 2: Prune the largest emitted top-level AND conjuncts first in
        # O(K log K) by tracking exact UTF-8 byte deltas via a max-heap.
        select_line_bytes = (
            1 + len(select_clause.encode("utf-8")) if select_clause is not None else 0
        )
        conj_byte_lens: dict[int, list[int]] = {}
        active_masks: dict[int, list[bool]] = {}
        filter_counts: dict[int, tuple[int, int, bool]] = {}
        for op_idx, (conjuncts, is_exact) in initial_filter_states.items():
            lengths = [len(c.encode("utf-8")) for c in conjuncts]
            conj_byte_lens[op_idx] = lengths
            active_masks[op_idx] = [True] * len(conjuncts)
            filter_counts[op_idx] = (len(lengths), sum(lengths), is_exact)

        emitted: set[int] = set(emitted_filter_indices)
        has_slice_after: set[int] = {
            op_idx
            for op_idx in initial_filter_states
            if any(
                isinstance(normalized_ops[j], Slice)
                for j in range(op_idx + 1, len(normalized_ops))
            )
        }

        # Max-heap ordered by (byte_len, op_idx, conj_idx) descending for
        # deterministic rightmost-first tie breaking matching `emitted`.
        heap: list[tuple[int, int, int]] = [
            (-conj_byte_lens[op_idx][conj_idx], -op_idx, -conj_idx)
            for op_idx in emitted
            for conj_idx in range(len(conj_byte_lens[op_idx]))
        ]
        heapq.heapify(heap)

        current_bytes = candidate_bytes
        while current_bytes > max_sql_bytes and emitted and heap:
            neg_len, neg_op_idx, neg_conj_idx = heapq.heappop(heap)
            op_idx = -neg_op_idx
            conj_idx = -neg_conj_idx
            if op_idx not in emitted:
                continue

            conj_len = -neg_len
            active_masks[op_idx][conj_idx] = False
            old_count, old_sum, was_exact = filter_counts[op_idx]
            new_count = old_count - 1
            new_sum = old_sum - conj_len
            filter_counts[op_idx] = (new_count, new_sum, False)

            if was_exact and op_idx in has_slice_after:
                # Changing a filter from exact -> inexact before a Slice can activate
                # partial_filter_barrier and cascade-withhold downstream slices/filters.
                current_bytes, emitted = _compute_active_lower_summary(
                    normalized_ops, filter_counts, inner_bytes, select_line_bytes
                )
            else:
                old_where_bytes = _where_clause_line_bytes(old_sum, old_count)
                new_where_bytes = _where_clause_line_bytes(new_sum, new_count)
                current_bytes -= old_where_bytes - new_where_bytes
                if new_count == 0:
                    emitted.discard(op_idx)

        if emitted and current_bytes <= max_sql_bytes:
            pruned_states: dict[int, tuple[tuple[str, ...], bool]] = {}
            for op_idx, (orig_conjuncts, orig_exact) in initial_filter_states.items():
                mask = active_masks[op_idx]
                _, _, is_exact = filter_counts[op_idx]
                kept = tuple(
                    c for c, keep in zip(orig_conjuncts, mask, strict=True) if keep
                )
                pruned_states[op_idx] = (kept, is_exact and orig_exact)
            pruned_candidate, _ = _lower_and_assemble(
                source=source,
                normalized_ops=normalized_ops,
                normalized_plan=normalized_plan,
                inner_query=inner_query,
                select_clause=select_clause,
                projected_columns=projected_columns,
                dummy_column=dummy_column,
                filter_states=pruned_states,
                degraded_reason="sql_length_pruned_conjuncts",
            )
            if len(pruned_candidate.sql.encode("utf-8")) <= max_sql_bytes:
                return pruned_candidate

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

    def __init__(
        self,
        arrow_client: ArrowQueryClient,
        *,
        max_sql_bytes: int | None = None,
    ) -> None:
        if max_sql_bytes is not None:
            max_sql_bytes = _validate_max_sql_bytes(max_sql_bytes)
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
        source_leaf, residual_ops = decompose_linear_plan(compiled.residual_plan)
        source = cast(QueryScan, source_leaf)

        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        if compiled.degraded_reason is not None or debug_enabled:
            sql_bytes = len(compiled.sql.encode("utf-8"))
            if compiled.degraded_reason is not None:
                logger.warning(
                    "Query pushdown degraded (%s): sql_bytes=%d, max_sql_bytes=%d",
                    compiled.degraded_reason,
                    sql_bytes,
                    active_max_sql_bytes,
                )
            if debug_enabled:
                total_columns = len(source.columns)
                if compiled.projected_columns is None:
                    projection_summary = f"{total_columns}/{total_columns}"
                elif len(compiled.projected_columns) == 0:
                    projection_summary = "0-col dummy"
                else:
                    projection_summary = (
                        f"{len(compiled.projected_columns)}/{total_columns}"
                    )
                where_bytes = sum(
                    len(c.encode("utf-8")) for c in compiled.where_clauses
                )
                logger.debug(
                    "Executing scan_query: maintain_order=%s, rewritten=%s, "
                    "projected=%s, where_clauses=%d, where_bytes=%d, "
                    "pushed_slices=%s, sql_bytes=%d, maximum_bytes_billed=%s, "
                    "degraded_reason=%s",
                    source.maintain_order,
                    compiled.rewritten,
                    projection_summary,
                    len(compiled.where_clauses),
                    where_bytes,
                    compiled.pushed_slices,
                    sql_bytes,
                    source.maximum_bytes_billed,
                    compiled.degraded_reason,
                )

        arrow_stream_exporter = self._arrow_client.read_query(
            compiled.sql,
            arrow_buffer_compression=source.arrow_buffer_compression,
            dry_run=False,
            maintain_order=source.maintain_order,
            use_query_cache=source.use_query_cache,
            maximum_bytes_billed=source.maximum_bytes_billed,
        )
        return execute_local_plan_batches(
            pl.scan_arrow_c_stream(arrow_stream_exporter),
            residual_ops,
            batch_size=batch_size,
            maintain_order=source.maintain_order,
        )


__all__ = [
    "MAX_BIGQUERY_LIMIT",
    "MAX_UNRESOLVED_SQL_BYTES",
    "CompiledQueryPlan",
    "DegradedReason",
    "QueryExecutor",
    "compile_query_plan",
    "has_non_comment_sql",
]
