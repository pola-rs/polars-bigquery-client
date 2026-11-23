"""Unit tests for `polars_bigquery.core.executors.query`."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import arrow_bigquery
import polars as pl
import pytest
from polars_bigquery.core.executors import (
    QueryExecutor,
    compile_query_plan,
)
from polars_bigquery.core.executors.query import (
    MAX_BIGQUERY_LIMIT,
    MAX_UNRESOLVED_SQL_BYTES,
)
from polars_bigquery.core.plan import (
    Filter,
    Project,
    QueryScan,
    Slice,
    TableScan,
    build_io_source_plan,
    decompose_linear_plan,
)
from polars_expressions_testing import unsupported


def _scan(
    sql: str = "SELECT a, b FROM t",
    *,
    columns: tuple[str, ...] = ("a", "b"),
    maintain_order: bool = False,
    explicit_schema: bool = False,
    compression: str = "lz4frame",
    use_query_cache: bool | None = None,
) -> QueryScan:
    return QueryScan(
        raw_query=sql,
        columns=columns,
        maintain_order=maintain_order,
        arrow_buffer_compression=compression,  # type: ignore[arg-type]
        use_query_cache=use_query_cache,
        explicit_schema=explicit_schema,
    )


def test_compile_query_plan_rejects_non_query_scan_leaf() -> None:
    table_scan = TableScan(
        table_ref=arrow_bigquery.BigQueryTableId("p", "d", "t"),
        schema=pl.Schema({"a": pl.Int64()}),
    )
    with pytest.raises(TypeError, match="expects a QueryScan leaf, got TableScan"):
        compile_query_plan(table_scan)


def test_compile_query_plan_maintain_order_true_bypasses_pushdown() -> None:
    raw = "\n  SELECT a, b FROM t ORDER BY a;  \n"
    scan = _scan(raw, maintain_order=True, explicit_schema=True)
    # Include a superseded intermediate Project to verify normalization still runs.
    plan = Project(
        input=Filter(
            input=Project(input=scan, columns=("a", "b")),
            predicate=pl.col("b") == "WA",
        ),
        columns=("a",),
    )
    compiled = compile_query_plan(plan, max_sql_bytes=5)
    assert compiled.sql == raw
    assert compiled.rewritten is False
    assert compiled.projected_columns is None
    assert compiled.where_clauses == ()
    assert compiled.pushed_slices == ()
    assert compiled.dummy_column is None
    assert compiled.degraded_reason is None
    _, residual_ops = decompose_linear_plan(compiled.residual_plan)
    assert [type(op) for op in residual_ops] == [Filter, Project]


def test_compile_query_plan_emission_matrix_dry_run() -> None:
    """Covers all 15 rows of the register_io_source SQL emission matrix (schema=None)."""
    base_raw = "SELECT a, b FROM `p.d.t`;"
    base_inner = "SELECT a, b FROM `p.d.t`"
    scan = _scan(base_raw, columns=("a", "b"), explicit_schema=False)
    p = pl.col("b") > 10
    p_sql = "(`b` > 10)"
    p_unpush = unsupported(pl.col("b")) > 10

    # 1. lf.collect() -> verbatim
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=None, n_rows=None)
    )
    assert c.sql == base_raw
    assert c.rewritten is False
    assert c.projected_columns is None

    # 2. lf.select(["b", "a"]) (all cols reordered) -> verbatim
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=["b", "a"], predicate=None, n_rows=None)
    )
    assert c.sql == base_raw
    assert c.rewritten is False

    # 3. lf.select(["a"]) (subset) -> |> SELECT `a`
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=["a"], predicate=None, n_rows=None)
    )
    assert c.sql == f"{base_inner}\n|> SELECT `a`"
    assert c.rewritten is True
    assert c.projected_columns == ("a",)

    # 4. lf.filter(p) (all cols, pushable) -> |> WHERE {sql(p)}
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=p, n_rows=None)
    )
    assert c.sql == f"{base_inner}\n|> WHERE {p_sql}"
    assert c.projected_columns is None
    assert c.where_clauses == (p_sql,)

    # 5. lf.filter(p_unpush) -> verbatim
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=p_unpush, n_rows=None)
    )
    assert c.sql == base_raw
    assert c.rewritten is False

    # 6. lf.filter(p).select(["a"]) -> |> WHERE {sql(p)}\n|> SELECT `a`, `b`
    scan3 = _scan(
        "SELECT a, b, c FROM `p.d.t`", columns=("a", "b", "c"), explicit_schema=False
    )
    c = compile_query_plan(
        build_io_source_plan(scan3, with_columns=["a"], predicate=p, n_rows=None)
    )
    assert c.sql == f"SELECT a, b, c FROM `p.d.t`\n|> WHERE {p_sql}\n|> SELECT `a`, `b`"
    assert c.projected_columns == ("a", "b")

    # 7. lf.filter(p).limit(10) -> register_io_source gets n_rows=None
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=p, n_rows=None)
    )
    assert c.sql == f"{base_inner}\n|> WHERE {p_sql}"
    assert c.pushed_slices == ()

    # 8. lf.limit(10) -> |> LIMIT 10
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=None, n_rows=10)
    )
    assert c.sql == f"{base_inner}\n|> LIMIT 10"
    assert c.pushed_slices == ((0, 10),)

    # Also test |> LIMIT 0
    c0 = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=None, n_rows=0)
    )
    assert c0.sql == f"{base_inner}\n|> LIMIT 0"
    assert c0.pushed_slices == ((0, 0),)

    # 9. lf.limit(10).select(["a"]) -> |> SELECT `a`\n|> LIMIT 10
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=["a"], predicate=None, n_rows=10)
    )
    assert c.sql == f"{base_inner}\n|> SELECT `a`\n|> LIMIT 10"

    # 10. lf.limit(10).filter(p) (pushable) -> |> WHERE {sql(p)}\n|> LIMIT 10
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=p, n_rows=10)
    )
    assert c.sql == f"{base_inner}\n|> WHERE {p_sql}\n|> LIMIT 10"

    # 11. lf.limit(10).filter(p).select(["a"]) (3 cols) -> |> WHERE\n|> SELECT `a`, `b`\n|> LIMIT 10
    c = compile_query_plan(
        build_io_source_plan(scan3, with_columns=["a"], predicate=p, n_rows=10)
    )
    assert (
        c.sql
        == f"SELECT a, b, c FROM `p.d.t`\n|> WHERE {p_sql}\n|> SELECT `a`, `b`\n|> LIMIT 10"
    )

    # 12. lf.limit(10).filter(p_unpush) -> |> LIMIT 10
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=p_unpush, n_rows=10)
    )
    assert c.sql == f"{base_inner}\n|> LIMIT 10"
    assert c.where_clauses == ()
    assert c.pushed_slices == ((0, 10),)

    # 13. lf.drop(pl.all()) -> |> SELECT 1 AS `_polars_bigquery_empty`
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=[], predicate=None, n_rows=None)
    )
    assert c.sql == f"{base_inner}\n|> SELECT 1 AS `_polars_bigquery_empty`"
    assert c.projected_columns == ()
    assert c.dummy_column == "_polars_bigquery_empty"

    # 14. lf.limit(10).drop(pl.all()) -> |> SELECT 1 AS `_polars_bigquery_empty`\n|> LIMIT 10
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=[], predicate=None, n_rows=10)
    )
    assert (
        c.sql == f"{base_inner}\n|> SELECT 1 AS `_polars_bigquery_empty`\n|> LIMIT 10"
    )
    assert c.projected_columns == ()
    assert c.dummy_column == "_polars_bigquery_empty"

    # 15. lf.filter(p).drop(pl.all()) -> |> WHERE {sql(p)}\n|> SELECT `b`
    c = compile_query_plan(
        build_io_source_plan(scan, with_columns=[], predicate=p, n_rows=None)
    )
    assert c.sql == f"{base_inner}\n|> WHERE {p_sql}\n|> SELECT `b`"
    assert c.projected_columns == ("b",)
    assert c.dummy_column is None


def test_compile_query_plan_canonical_column_order_for_cache_sharing() -> None:
    scan = _scan("SELECT a, b, c FROM t", columns=("a", "b", "c"))
    p_ab = compile_query_plan(
        build_io_source_plan(scan, with_columns=["a", "b"], predicate=None, n_rows=None)
    )
    p_ba = compile_query_plan(
        build_io_source_plan(scan, with_columns=["b", "a"], predicate=None, n_rows=None)
    )
    assert p_ab.sql == p_ba.sql == "SELECT a, b, c FROM t\n|> SELECT `a`, `b`"


def test_compile_query_plan_explicit_schema_always_emits_select() -> None:
    scan = _scan(
        "SELECT * FROM wide_table",
        columns=("a", "b"),
        explicit_schema=True,
    )
    # Plain lf.collect()
    c1 = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=None, n_rows=None)
    )
    assert c1.sql == "SELECT * FROM wide_table\n|> SELECT `a`, `b`"
    assert c1.projected_columns == ("a", "b")
    assert c1.rewritten is True

    # No-Project plan also emits |> SELECT
    c_bare = compile_query_plan(scan)
    assert c_bare.sql == "SELECT * FROM wide_table\n|> SELECT `a`, `b`"

    # Reordered lf.select(["b", "a"]) emits canonical schema declaration order
    c2 = compile_query_plan(
        build_io_source_plan(scan, with_columns=["b", "a"], predicate=None, n_rows=None)
    )
    assert c2.sql == c1.sql

    # Filter on all explicit columns
    c3 = compile_query_plan(
        build_io_source_plan(
            scan, with_columns=None, predicate=pl.col("a") == 1, n_rows=None
        )
    )
    assert c3.sql == "SELECT * FROM wide_table\n|> WHERE (`a` = 1)\n|> SELECT `a`, `b`"

    # Limit on all explicit columns
    c4 = compile_query_plan(
        build_io_source_plan(scan, with_columns=None, predicate=None, n_rows=10)
    )
    assert c4.sql == "SELECT * FROM wide_table\n|> SELECT `a`, `b`\n|> LIMIT 10"

    # Under maintain_order=True, explicit schema still runs verbatim
    ordered = _scan(
        "SELECT * FROM wide_table",
        columns=("a", "b"),
        maintain_order=True,
        explicit_schema=True,
    )
    c5 = compile_query_plan(
        build_io_source_plan(ordered, with_columns=None, predicate=None, n_rows=None)
    )
    assert c5.sql == "SELECT * FROM wide_table"
    assert c5.rewritten is False


def test_compile_query_plan_general_plan_orderings() -> None:
    scan = _scan("SELECT a, b, c FROM t", columns=("a", "b", "c"))
    exact_p = pl.col("b") == 1
    partial_p = (pl.col("b") == 1) & (unsupported(pl.col("a")) == 2)
    unpush_p = unsupported(pl.col("b")) == 1
    q = pl.col("a") == 3

    # Filter(exact) -> Slice(0, 10) -> Project(["a"])
    c = compile_query_plan(
        Project(
            input=Slice(
                input=Filter(input=scan, predicate=exact_p), offset=0, length=10
            ),
            columns=("a",),
        )
    )
    assert (
        c.sql
        == "SELECT a, b, c FROM t\n|> WHERE (`b` = 1)\n|> SELECT `a`, `b`\n|> LIMIT 10"
    )
    assert c.pushed_slices == ((0, 10),)

    # Filter(partial) -> Slice(0, 10) -> Project(["a"]) -> NO SQL LIMIT
    c = compile_query_plan(
        Project(
            input=Slice(
                input=Filter(input=scan, predicate=partial_p), offset=0, length=10
            ),
            columns=("a",),
        )
    )
    assert c.sql == "SELECT a, b, c FROM t\n|> WHERE (`b` = 1)\n|> SELECT `a`, `b`"
    assert c.pushed_slices == ()

    # Filter(exact) -> Slice(5, 10) -> Project(["a"]) -> offset cleared in residual
    c = compile_query_plan(
        Project(
            input=Slice(
                input=Filter(input=scan, predicate=exact_p), offset=5, length=10
            ),
            columns=("a",),
        )
    )
    assert (
        c.sql
        == "SELECT a, b, c FROM t\n|> WHERE (`b` = 1)\n|> SELECT `a`, `b`\n|> LIMIT 10 OFFSET 5"
    )
    assert c.pushed_slices == ((5, 10),)
    _, res_ops = decompose_linear_plan(c.residual_plan)
    assert isinstance(res_ops[1], Slice)
    assert res_ops[1].offset == 0
    assert res_ops[1].length == 10

    # Slice(5, 10) -> Filter(exact) -> Project(["a"]) -> offset cleared
    c = compile_query_plan(
        Project(
            input=Filter(
                input=Slice(input=scan, offset=5, length=10), predicate=exact_p
            ),
            columns=("a",),
        )
    )
    assert (
        c.sql
        == "SELECT a, b, c FROM t\n|> LIMIT 10 OFFSET 5\n|> WHERE (`b` = 1)\n|> SELECT `a`, `b`"
    )
    _, res_ops = decompose_linear_plan(c.residual_plan)
    assert isinstance(res_ops[0], Slice)
    assert res_ops[0].offset == 0
    assert res_ops[0].length == 10

    # Slice(0, 10) -> Slice(5, 10) -> Filter(exact) -> flush LIMIT 10 before LIMIT 10 OFFSET 5
    c = compile_query_plan(
        Filter(
            input=Slice(
                input=Slice(input=scan, offset=0, length=10),
                offset=5,
                length=10,
            ),
            predicate=exact_p,
        )
    )
    assert (
        c.sql
        == "SELECT a, b, c FROM t\n|> LIMIT 10\n|> LIMIT 10 OFFSET 5\n|> WHERE (`b` = 1)"
    )
    assert c.pushed_slices == ((0, 10), (5, 10))

    # Slice(0, 10) -> Filter(partial) -> Filter(exact) -> flush after partial before exact
    c = compile_query_plan(
        Filter(
            input=Filter(
                input=Slice(input=scan, offset=0, length=10),
                predicate=partial_p,
            ),
            predicate=q,
        )
    )
    assert (
        c.sql
        == "SELECT a, b, c FROM t\n|> WHERE (`b` = 1)\n|> LIMIT 10\n|> WHERE (`a` = 3)"
    )

    # Slice(0, 10) -> Filter(unpushable) -> Filter(exact) -> flush LIMIT 10 before WHERE q
    c = compile_query_plan(
        Filter(
            input=Filter(
                input=Slice(input=scan, offset=0, length=10),
                predicate=unpush_p,
            ),
            predicate=q,
        )
    )
    assert c.sql == "SELECT a, b, c FROM t\n|> LIMIT 10\n|> WHERE (`a` = 3)"

    # Filter(partial) -> Filter(exact) -> Slice(2, 5) -> Filter(q) -> Slice(0, 2)
    # Adjacent Filter emitted while partial_filter_barrier=True; unpushed Slice blocks upper ops.
    c = compile_query_plan(
        Slice(
            input=Filter(
                input=Slice(
                    input=Filter(
                        input=Filter(input=scan, predicate=partial_p),
                        predicate=q,
                    ),
                    offset=2,
                    length=5,
                ),
                predicate=pl.col("c") == 9,
            ),
            offset=0,
            length=2,
        )
    )
    assert c.sql == "SELECT a, b, c FROM t\n|> WHERE (`b` = 1)\n|> WHERE (`a` = 3)"
    assert c.pushed_slices == ()
    _, res_ops = decompose_linear_plan(c.residual_plan)
    assert isinstance(res_ops[2], Slice)
    assert res_ops[2].offset == 2
    assert res_ops[2].length == 5
    assert isinstance(res_ops[4], Slice)
    assert res_ops[4].offset == 0
    assert res_ops[4].length == 2


def test_compile_query_plan_multi_project_collapse_no_column_error() -> None:
    scan = _scan("SELECT a, b, c FROM t", columns=("a", "b", "c"))
    plan = Project(
        input=Filter(
            input=Project(input=scan, columns=("a", "b")),
            predicate=pl.col("b") == 1,
        ),
        columns=("a",),
    )
    c = compile_query_plan(plan)
    assert c.sql == "SELECT a, b, c FROM t\n|> WHERE (`b` = 1)\n|> SELECT `a`, `b`"
    _, res_ops = decompose_linear_plan(c.residual_plan)
    assert [type(op) for op in res_ops] == [Filter, Project]


def test_compile_query_plan_out_of_range_slices_kept_local() -> None:
    scan = _scan("SELECT a FROM t", columns=("a",))
    u64_max = (1 << 64) - 1

    # Out-of-range length (> MAX_BIGQUERY_LIMIT)
    c_len = compile_query_plan(
        Filter(
            input=Slice(input=scan, offset=0, length=u64_max),
            predicate=pl.col("a") == 1,
        )
    )
    assert c_len.sql == "SELECT a FROM t"
    assert c_len.rewritten is False
    assert c_len.pushed_slices == ()
    _, ops_len = decompose_linear_plan(c_len.residual_plan)
    assert ops_len[0] == Slice(input=scan, offset=0, length=u64_max)

    # Out-of-range offset (> MAX_BIGQUERY_LIMIT)
    c_off = compile_query_plan(
        Slice(input=scan, offset=MAX_BIGQUERY_LIMIT + 1, length=5)
    )
    assert c_off.sql == "SELECT a FROM t"
    assert c_off.rewritten is False
    _, ops_off = decompose_linear_plan(c_off.residual_plan)
    assert ops_off[0] == Slice(input=scan, offset=MAX_BIGQUERY_LIMIT + 1, length=5)

    # Exactly MAX_BIGQUERY_LIMIT fits INT64 and IS pushed
    c_exact = compile_query_plan(
        Slice(input=scan, offset=MAX_BIGQUERY_LIMIT, length=MAX_BIGQUERY_LIMIT)
    )
    assert (
        c_exact.sql
        == f"SELECT a FROM t\n|> LIMIT {MAX_BIGQUERY_LIMIT} OFFSET {MAX_BIGQUERY_LIMIT}"
    )

    # maintain_order=True with huge slice works without ValueError
    ordered_scan = _scan("SELECT a FROM t", columns=("a",), maintain_order=True)
    c_ord = compile_query_plan(
        Slice(input=ordered_scan, offset=u64_max, length=u64_max)
    )
    assert c_ord.rewritten is False


def test_compile_query_plan_whitespace_semicolons_and_user_pipe() -> None:
    # Preserves leading newlines/indentation so [line:col] stays unshifted.
    triple = "\n\n  SELECT a, b\n  FROM `p.d.t`;  \n"
    scan = _scan(triple, columns=("a", "b"))
    c = compile_query_plan(Project(input=scan, columns=("a",)))
    assert c.sql == "\n\n  SELECT a, b\n  FROM `p.d.t`\n|> SELECT `a`"

    # Double trailing semicolon strips only ONE semicolon so BigQuery fails it.
    scan_double = _scan("SELECT a, b FROM t;;", columns=("a", "b"))
    c_double = compile_query_plan(Project(input=scan_double, columns=("a",)))
    assert c_double.sql == "SELECT a, b FROM t;\n|> SELECT `a`"

    # User query already containing pipe |> WHERE must not confuse last_where_idx!
    user_pipe = "FROM t\n|> WHERE b > 0\n|> LIMIT 5"
    scan_pipe = _scan(user_pipe, columns=("a", "b"))
    c_pipe = compile_query_plan(
        Project(input=Slice(input=scan_pipe, offset=0, length=2), columns=("a",))
    )
    assert c_pipe.sql == f"{user_pipe}\n|> SELECT `a`\n|> LIMIT 2"


def test_compile_query_plan_trivial_true_preserves_exactness() -> None:
    scan = _scan("SELECT a FROM t", columns=("a",))
    plan = Slice(
        input=Filter(input=scan, predicate=pl.lit(True)),
        offset=0,
        length=5,
    )
    c = compile_query_plan(plan)
    assert c.sql == "SELECT a FROM t\n|> LIMIT 5"
    assert c.where_clauses == ()
    assert c.pushed_slices == ((0, 5),)


def test_compile_query_plan_dummy_column_collision_safety() -> None:
    scan1 = _scan(
        "SELECT `_POLARS_BIGQUERY_EMPTY` FROM t", columns=("_POLARS_BIGQUERY_EMPTY",)
    )
    c1 = compile_query_plan(Project(input=scan1, columns=()))
    assert c1.dummy_column == "_polars_bigquery_empty_1"
    assert c1.sql.endswith("|> SELECT 1 AS `_polars_bigquery_empty_1`")

    scan2 = _scan(
        "SELECT * FROM t",
        columns=(
            "_polars_bigquery_empty",
            "_POLARS_BIGQUERY_EMPTY_1",
            "_polars_bigquery_empty_2",
        ),
    )
    c2 = compile_query_plan(Project(input=scan2, columns=()))
    assert c2.dummy_column == "_polars_bigquery_empty_3"


def test_compile_query_plan_disallowed_column_identifier_raises() -> None:
    scan = _scan("SELECT 1", columns=("bad`col", "unused"))
    with pytest.raises(ValueError, match="Invalid BigQuery column name"):
        compile_query_plan(Project(input=scan, columns=("bad`col",)))

    # Flexible valid column names with spaces are allowed.
    flex_scan = _scan(
        "SELECT 1 AS `first name`, 2 AS unused",
        columns=("first name", "unused"),
    )
    c = compile_query_plan(Project(input=flex_scan, columns=("first name",)))
    assert c.sql == "SELECT 1 AS `first name`, 2 AS unused\n|> SELECT `first name`"


def test_compile_query_plan_length_guard_tiers_and_utf8() -> None:
    # Unrewritten query > max_sql_bytes bypasses length guard with degraded_reason=None
    huge_raw = "SELECT a, b FROM `" + "x" * 100 + "`"
    scan_huge = _scan(huge_raw, columns=("a", "b"))
    c_unrewritten = compile_query_plan(scan_huge, max_sql_bytes=20)
    assert c_unrewritten.sql == huge_raw
    assert c_unrewritten.rewritten is False
    assert c_unrewritten.degraded_reason is None

    # Tier 1: multi-byte UTF-8 boundary check (é is 2 bytes, 🦀 is 4 bytes)
    scan = _scan("SELECT a, b, c FROM t", columns=("a", "b", "c"))
    pred_utf8 = pl.col("a") == "ééé"
    c_fit = compile_query_plan(Filter(input=scan, predicate=pred_utf8))
    exact_len = len(c_fit.sql.encode("utf-8"))
    assert exact_len > len(c_fit.sql)  # confirm multi-byte UTF-8 inflation
    assert (
        compile_query_plan(
            Filter(input=scan, predicate=pred_utf8), max_sql_bytes=exact_len
        ).degraded_reason
        is None
    )
    # One byte smaller degrades to Tier 4 verbatim fallback (no |> SELECT or |> LIMIT left)
    c_over = compile_query_plan(
        Filter(input=scan, predicate=pred_utf8), max_sql_bytes=exact_len - 1
    )
    assert c_over.degraded_reason == "sql_length_skipped_pushdown"
    assert c_over.sql == "SELECT a, b, c FROM t"
    assert c_over.rewritten is False

    # Tier 2: Prunes largest conjunct, keeping smaller sibling predicate
    pred_combo = (pl.col("a") == 1) & (
        pl.col("b").is_in(["longvalue_" + str(i) for i in range(10)])
    )
    full_combo = compile_query_plan(
        Project(input=Filter(input=scan, predicate=pred_combo), columns=("a",))
    )
    small_only = "SELECT a, b, c FROM t\n|> WHERE (`a` = 1)\n|> SELECT `a`, `b`"
    c_tier2 = compile_query_plan(
        Project(input=Filter(input=scan, predicate=pred_combo), columns=("a",)),
        max_sql_bytes=len(small_only.encode("utf-8")) + 5,
    )
    assert c_tier2.degraded_reason == "sql_length_pruned_conjuncts"
    assert c_tier2.sql == small_only
    assert c_tier2.where_clauses == ("(`a` = 1)",)
    assert len(full_combo.sql.encode("utf-8")) > len(small_only.encode("utf-8"))

    # Tier 2 tie-breaking: equal UTF-8 byte lengths drop rightmost conjunct first
    pred_tie = (pl.col("a") == "xx") & (pl.col("b") == "yy")
    keep_left = "SELECT a, b, c FROM t\n|> WHERE (`a` = 'xx')"
    c_tie = compile_query_plan(
        Filter(input=scan, predicate=pred_tie),
        max_sql_bytes=len(keep_left.encode("utf-8")),
    )
    assert c_tie.degraded_reason == "sql_length_pruned_conjuncts"
    assert c_tie.where_clauses == ("(`a` = 'xx')",)

    # Tier 2 cross-filter tie-breaking: drops later Filter stage first
    plan_two_filters = Filter(
        input=Filter(input=scan, predicate=pl.col("a") == "xx"),
        predicate=pl.col("b") == "yy",
    )
    c_two_tie = compile_query_plan(
        plan_two_filters,
        max_sql_bytes=len(keep_left.encode("utf-8")),
    )
    assert c_two_tie.degraded_reason == "sql_length_pruned_conjuncts"
    assert c_two_tie.where_clauses == ("(`a` = 'xx')",)

    # Tier 2 re-lowering: Filter(c1_small & c1_huge) -> Slice(5, 10) -> Filter(c2_small)
    # Dropping c1_huge marks Filter 1 partial, withholding BOTH Slice(5, 10) (restoring offset=5!)
    # and upper Filter(c2_small).
    plan_slice_restore = Filter(
        input=Slice(
            input=Filter(
                input=scan,
                predicate=(pl.col("a") == 1) & (pl.col("b") == "x" * 60),
            ),
            offset=5,
            length=10,
        ),
        predicate=pl.col("c") == 2,
    )
    c_restored = compile_query_plan(
        plan_slice_restore,
        max_sql_bytes=len(b"SELECT a, b, c FROM t\n|> WHERE (`a` = 1)") + 5,
    )
    assert c_restored.degraded_reason == "sql_length_pruned_conjuncts"
    assert c_restored.sql == "SELECT a, b, c FROM t\n|> WHERE (`a` = 1)"
    assert c_restored.pushed_slices == ()
    _, restored_ops = decompose_linear_plan(c_restored.residual_plan)
    assert isinstance(restored_ops[1], Slice)
    assert restored_ops[1].offset == 5
    assert restored_ops[1].length == 10

    # Tier 2 exits immediately to Tier 3 when dropping the first Filter leaves
    # candidate.where_clauses empty (Filter(c1_huge) -> Slice(0, 10) -> Filter(c2_small))
    plan_cascade_drop = Project(
        input=Filter(
            input=Slice(
                input=Filter(input=scan, predicate=pl.col("b") == "x" * 60),
                offset=0,
                length=10,
            ),
            predicate=pl.col("a") == 1,
        ),
        columns=("a",),
    )
    # Budget fits `SELECT a, b, c FROM t\n|> SELECT `a`, `b`` but not with WHERE b == xxx
    expected_tier3_sql = "SELECT a, b, c FROM t\n|> SELECT `a`, `b`"
    c_cascade = compile_query_plan(
        plan_cascade_drop,
        max_sql_bytes=len(expected_tier3_sql.encode("utf-8")) + 2,
    )
    assert c_cascade.degraded_reason == "sql_length_dropped_where"
    assert c_cascade.sql == expected_tier3_sql
    assert c_cascade.where_clauses == ()
    assert c_cascade.pushed_slices == ()

    # Tier 3: preserves trivial pl.lit(True) exactness and retains pre-filter Slice!
    plan_trivial_before_slice = Filter(
        input=Slice(
            input=Filter(input=scan, predicate=pl.lit(True)),
            offset=0,
            length=10,
        ),
        predicate=pl.col("b") == "y" * 60,
    )
    expected_with_limit = "SELECT a, b, c FROM t\n|> LIMIT 10"
    c_t3_limit = compile_query_plan(
        plan_trivial_before_slice,
        max_sql_bytes=len(expected_with_limit.encode("utf-8")),
    )
    assert c_t3_limit.degraded_reason == "sql_length_dropped_where"
    assert c_t3_limit.sql == expected_with_limit
    assert c_t3_limit.pushed_slices == ((0, 10),)

    # Tier 3 falls through to Tier 4 when remaining |> SELECT / |> LIMIT still exceeds budget
    c_t4_from_t3 = compile_query_plan(
        plan_trivial_before_slice,
        max_sql_bytes=len(expected_with_limit.encode("utf-8")) - 1,
    )
    assert c_t4_from_t3.degraded_reason == "sql_length_skipped_pushdown"
    assert c_t4_from_t3.sql == "SELECT a, b, c FROM t"
    assert c_t4_from_t3.rewritten is False

    # Tier 4 directly when rewritten query has NO |> WHERE clauses at all (only |> SELECT)
    c_t4_no_where = compile_query_plan(
        Project(input=scan, columns=("a",)),
        max_sql_bytes=10,
    )
    assert c_t4_no_where.degraded_reason == "sql_length_skipped_pushdown"
    assert c_t4_no_where.sql == "SELECT a, b, c FROM t"
    assert c_t4_no_where.rewritten is False


def test_query_executor_execute_logging_and_arguments(caplog) -> None:
    mock_arrow_client = MagicMock()
    mock_arrow_client.read_query.return_value = MagicMock(name="exporter")
    source_lf = pl.LazyFrame(
        {"a": [1, 2, 3], "b": ["secret_keep", "secret_drop", "secret_keep"]}
    )

    # 1. Standard execution with partial projection (no warning, DEBUG logged)
    scan = _scan(
        "SELECT a, b, c FROM secret_tbl",
        columns=("a", "b", "c"),
        maintain_order=False,
        compression="zstd",
        use_query_cache=False,
    )
    plan = build_io_source_plan(
        scan,
        with_columns=["a"],
        predicate=pl.col("b") == "secret_keep",
        n_rows=None,
    )
    executor = QueryExecutor(mock_arrow_client)
    with (
        caplog.at_level(logging.DEBUG, logger="polars_bigquery.core.executors.query"),
        patch("polars.scan_arrow_c_stream", return_value=source_lf),
    ):
        batches = list(executor.execute(plan, batch_size=2))

    assert sorted(pl.concat(batches)["a"].to_list()) == [1, 3]
    mock_arrow_client.read_query.assert_called_once_with(
        "SELECT a, b, c FROM secret_tbl\n|> WHERE (`b` = 'secret_keep')\n|> SELECT `a`, `b`",
        arrow_buffer_compression="zstd",
        dry_run=False,
        maintain_order=False,
        use_query_cache=False,
        maximum_bytes_billed=None,
    )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []
    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debug_msgs) == 1
    assert "projected=2/3" in debug_msgs[0]
    assert "secret_tbl" not in debug_msgs[0]
    assert "secret_keep" not in debug_msgs[0]

    # 2. All-column unpruned execution logs total/total
    caplog.clear()
    ordered_scan = _scan(
        "SELECT a, b FROM secret_tbl", columns=("a", "b"), maintain_order=True
    )
    with (
        caplog.at_level(logging.DEBUG, logger="polars_bigquery.core.executors.query"),
        patch("polars.scan_arrow_c_stream", return_value=source_lf),
    ):
        list(executor.execute(ordered_scan))
    assert any("projected=2/2" in r.message for r in caplog.records)

    # 3. 0-column dummy projection logs "0-col dummy"
    caplog.clear()
    dummy_lf = pl.LazyFrame({"_polars_bigquery_empty": [1, 1]})
    with (
        caplog.at_level(logging.DEBUG, logger="polars_bigquery.core.executors.query"),
        patch("polars.scan_arrow_c_stream", return_value=dummy_lf),
    ):
        list(executor.execute(Project(input=scan, columns=())))
    assert any("projected=0-col dummy" in r.message for r in caplog.records)

    # 4. Degraded execution emits a single WARNING naming degraded_reason, sql_bytes, and max_sql_bytes
    caplog.clear()
    tiny_executor = QueryExecutor(mock_arrow_client, max_sql_bytes=45)
    with (
        caplog.at_level(logging.DEBUG, logger="polars_bigquery.core.executors.query"),
        patch("polars.scan_arrow_c_stream", return_value=source_lf),
    ):
        list(tiny_executor.execute(plan))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    warn_msg = warnings[0].message
    assert "sql_length_skipped_pushdown" in warn_msg
    assert "max_sql_bytes=45" in warn_msg
    assert f"sql_bytes={len(scan.raw_query.encode('utf-8'))}" in warn_msg
    assert "secret_tbl" not in warn_msg
    assert "secret_keep" not in warn_msg
    assert MAX_UNRESOLVED_SQL_BYTES == 1_000_000


def test_query_executor_forwards_maintain_order_to_collect_batches() -> None:
    mock_arrow_client = MagicMock()
    mock_arrow_client.read_query.return_value = MagicMock(name="exporter")
    mock_lf = MagicMock()
    mock_lf.filter.return_value = mock_lf
    mock_lf.select.return_value = mock_lf
    mock_lf.collect_batches.return_value = iter([pl.DataFrame({"a": [1]})])

    executor = QueryExecutor(mock_arrow_client)
    for maintain_order in (True, False):
        mock_lf.collect_batches.reset_mock()
        scan = _scan("SELECT a FROM t", columns=("a",), maintain_order=maintain_order)
        with patch("polars.scan_arrow_c_stream", return_value=mock_lf):
            list(executor.execute(scan, batch_size=128))
        mock_lf.collect_batches.assert_called_once_with(
            chunk_size=128,
            maintain_order=maintain_order,
        )


def test_compile_query_plan_validates_max_sql_bytes() -> None:
    scan = _scan("SELECT a FROM t", columns=("a",))
    for bad in (True, False, 1.5, "100"):
        with pytest.raises(TypeError, match="positive int"):
            compile_query_plan(scan, max_sql_bytes=bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="positive int"):
            QueryExecutor(MagicMock(), max_sql_bytes=bad)  # type: ignore[arg-type]

    for non_pos in (0, -1, -100):
        with pytest.raises(ValueError, match="must be positive"):
            compile_query_plan(scan, max_sql_bytes=non_pos)
        with pytest.raises(ValueError, match="must be positive"):
            QueryExecutor(MagicMock(), max_sql_bytes=non_pos)


def test_compile_query_plan_trailing_semicolon_with_comments_and_literals() -> None:
    cases = [
        # Trailing comments placed after a single `;` are preserved (e.g.
        # SQLCommenter / dbt tracing tags) while removing only the bare `;`.
        ("SELECT a, b FROM t; -- trailing note", "SELECT a, b FROM t -- trailing note"),
        ("SELECT a, b FROM t; # hash comment", "SELECT a, b FROM t # hash comment"),
        (
            "SELECT a, b FROM t; /* trace_id: abc */",
            "SELECT a, b FROM t /* trace_id: abc */",
        ),
        (
            "SELECT a, b FROM t;\n-- next line comment\n",
            "SELECT a, b FROM t\n-- next line comment",
        ),
        (
            "SELECT a, b FROM t; /* a */ -- b",
            "SELECT a, b FROM t /* a */ -- b",
        ),
        # Classic CR-only (`\r`) line-comment terminators.
        ("SELECT a, b FROM t\r-- note\r;", "SELECT a, b FROM t\r-- note"),
        ("SELECT a, b FROM t\r# note\r;", "SELECT a, b FROM t\r# note"),
        # Preserve invalid double semicolon so BigQuery still fails it.
        ("SELECT a, b FROM t;; -- note", "SELECT a, b FROM t; -- note"),
        ("SELECT a, b FROM t; /* a */;", "SELECT a, b FROM t; /* a */"),
        # Unterminated block comment/string must NOT strip a preceding `;`
        # (prevents silently swallowing syntax errors).
        (
            "SELECT a, b FROM t; /* unterminated",
            "SELECT a, b FROM t; /* unterminated",
        ),
        (
            "SELECT a, b FROM t; /*/",
            "SELECT a, b FROM t; /*/",
        ),
        # Semicolon INSIDE a line comment must NOT be stripped.
        (
            "SELECT a, b FROM t -- keep semi;",
            "SELECT a, b FROM t -- keep semi;",
        ),
        # Semicolon INSIDE string/raw-string/backtick literals must NOT be stripped.
        ("SELECT 'x;' AS a, b FROM t", "SELECT 'x;' AS a, b FROM t"),
        ('SELECT r"x;" AS a, b FROM t', 'SELECT r"x;" AS a, b FROM t'),
        ("SELECT '''x;''' AS a, b FROM t;", "SELECT '''x;''' AS a, b FROM t"),
        # Raw strings with backslash-escaped quotes keep literal boundaries in sync.
        (r"SELECT r'a\'b' AS a, b FROM t;", r"SELECT r'a\'b' AS a, b FROM t"),
        (r'SELECT r"a\"b" AS a, b FROM t;', r'SELECT r"a\"b" AS a, b FROM t'),
        (r"SELECT r'a\\' AS a, b FROM t;", r"SELECT r'a\\' AS a, b FROM t"),
        (
            r"SELECT rb'''a\'''b''' AS a, b FROM t;",
            r"SELECT rb'''a\'''b''' AS a, b FROM t",
        ),
        (
            r"SELECT a, b FROM t WHERE a = r'x\'y' AND b = '/*;*/;   -- comment'",
            r"SELECT a, b FROM t WHERE a = r'x\'y' AND b = '/*;*/;   -- comment'",
        ),
        # Backtick identifiers with escaped backslash / backtick.
        (r"SELECT `a\`b` AS a, b FROM t;", r"SELECT `a\`b` AS a, b FROM t"),
        (r"SELECT `a\\` AS a, b FROM t;", r"SELECT `a\\` AS a, b FROM t"),
        # Leading newlines/indentation stay unshifted.
        ("\n\n  SELECT a, b FROM t; -- note\n", "\n\n  SELECT a, b FROM t -- note"),
        # CREATE TEMP FUNCTION statement retains its internal semicolon.
        (
            "CREATE TEMP FUNCTION f(x INT64) AS (x + 1); SELECT f(1) AS a, 2 AS b; -- end",
            "CREATE TEMP FUNCTION f(x INT64) AS (x + 1); SELECT f(1) AS a, 2 AS b -- end",
        ),
    ]
    for raw_sql, expected_inner in cases:
        scan = _scan(raw_sql, columns=("a", "b"))
        compiled = compile_query_plan(Project(input=scan, columns=("a",)))
        assert compiled.sql == f"{expected_inner}\n|> SELECT `a`", (
            f"Unexpected rewrite for {raw_sql!r}: {compiled.sql!r}"
        )


def test_has_non_comment_sql_rejects_empty_semicolons_and_comments_only() -> None:
    from polars_bigquery.core.executors import has_non_comment_sql

    for empty in (
        "",
        "   \n\t ",
        ";",
        " ; ; ",
        ";;;",
        "-- only a comment",
        "# hash comment\n;",
        "/* block comment */ ; -- trailing",
        "/* unclosed comment",
    ):
        assert has_non_comment_sql(empty) is False, empty

    assert has_non_comment_sql("-- header\nSELECT 1") is True
    assert has_non_comment_sql("/* note */ SELECT r'\\\\'") is True
    # Unterminated raw/quoted literal or backtick is still non-empty SQL so BigQuery
    # reports the syntax error rather than claiming query is empty.
    assert has_non_comment_sql("SELECT 'unterminated\\") is True
    assert has_non_comment_sql("SELECT `unterminated") is True


def test_sql_trailing_semicolon_scanner_scales_linearly() -> None:
    import time

    from polars_bigquery.core.executors.query import _strip_single_trailing_semicolon

    adversarial_cases = [
        "SELECT '" + ("\\\\" * 100_000) + "a';",
        "SELECT 1 - " + ("1 - " * 100_000) + "1;",
        "SELECT a FROM t WHERE x IN (" + ",".join(["123"] * 100_000) + "); -- tail",
    ]
    for sql in adversarial_cases:
        t0 = time.perf_counter()
        stripped = _strip_single_trailing_semicolon(sql)
        elapsed = time.perf_counter() - t0
        assert not stripped.endswith(";")
        assert elapsed < 0.5, f"SQL scanner took {elapsed:.3f}s on len={len(sql)}"


def test_compile_query_plan_inner_query_fast_path_and_large_conjunct_scaling() -> None:
    import time

    # 1. Fast-path directly to Tier 4 when inner_query >= max_sql_bytes
    scan_big = _scan("SELECT a FROM " + "x" * 100, columns=("a",))
    expr = pl.all_horizontal(*[pl.col("a") == i for i in range(100)])
    fast = compile_query_plan(Filter(input=scan_big, predicate=expr), max_sql_bytes=50)
    assert fast.degraded_reason == "sql_length_skipped_pushdown"
    assert fast.rewritten is False
    assert fast.sql == scan_big.raw_query

    # 2. 2,000 conjuncts (~850 KB -> 100 KB) must prune in O(K log K), well under 1 s!
    many_expr = pl.all_horizontal(
        *[pl.col("a") == f"val_{i:04d}_" + "x" * 400 for i in range(2000)]
    )
    scan = _scan("SELECT a FROM t", columns=("a",))
    plan = Filter(input=scan, predicate=many_expr)
    t0 = time.perf_counter()
    compiled = compile_query_plan(plan, max_sql_bytes=100_000)
    elapsed = time.perf_counter() - t0
    assert compiled.degraded_reason == "sql_length_pruned_conjuncts"
    assert len(compiled.sql.encode("utf-8")) <= 100_000
    assert len(compiled.sql.encode("utf-8")) > 90_000
    assert elapsed < 2.0, f"Tier 2 pruning regressed to slow runtime: {elapsed:.2f}s"


def test_query_executor_handles_huge_slice() -> None:
    mock_arrow_client = MagicMock()
    mock_arrow_client.read_query.return_value = MagicMock(name="exporter")
    scan = QueryScan(
        raw_query="SELECT a FROM t",
        columns=("a",),
        maintain_order=False,
    )
    plan = Slice(input=scan, offset=0, length=(1 << 64) - 1)
    executor = QueryExecutor(mock_arrow_client)
    with patch(
        "polars.scan_arrow_c_stream", return_value=pl.LazyFrame({"a": [1, 2, 3]})
    ):
        batches = list(executor.execute(plan))
    assert sorted(pl.concat(batches)["a"].to_list()) == [1, 2, 3]
    mock_arrow_client.read_query.assert_called_once_with(
        "SELECT a FROM t",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=False,
        use_query_cache=None,
        maximum_bytes_billed=None,
    )


def test_query_executor_forwards_maximum_bytes_billed(caplog) -> None:
    import logging

    mock_arrow_client = MagicMock()
    mock_arrow_client.read_query.return_value = MagicMock(name="exporter")
    scan = QueryScan(
        raw_query="SELECT a FROM t",
        columns=("a",),
        maintain_order=False,
        maximum_bytes_billed=10_000_000,
    )
    executor = QueryExecutor(mock_arrow_client)
    with (
        caplog.at_level(logging.DEBUG, logger="polars_bigquery.core.executors.query"),
        patch("polars.scan_arrow_c_stream", return_value=pl.LazyFrame({"a": [1]})),
    ):
        batches = list(executor.execute(scan))
    assert pl.concat(batches).to_dict(as_series=False) == {"a": [1]}
    mock_arrow_client.read_query.assert_called_once_with(
        "SELECT a FROM t",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=False,
        use_query_cache=None,
        maximum_bytes_billed=10_000_000,
    )
    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debug_msgs) == 1
    assert "maximum_bytes_billed=10000000" in debug_msgs[0]


def test_query_executor_closes_exporter_on_post_export_error() -> None:
    close_calls = 0

    class _Exporter:
        def __arrow_c_stream__(self, requested_schema=None):
            return pl.DataFrame({"a": [1]}).__arrow_c_stream__(
                requested_schema=requested_schema
            )

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    mock_arrow_client = MagicMock()
    mock_arrow_client.read_query.return_value = _Exporter()
    executor = QueryExecutor(mock_arrow_client)
    scan = _scan("SELECT a FROM t", columns=("a",))

    with (
        patch(
            "polars_bigquery.core.executors.query.execute_local_plan_batches",
            side_effect=RuntimeError("boom after stream export"),
        ),
        pytest.raises(RuntimeError, match="boom after stream export") as exc_info,
    ):
        executor.execute(scan)

    assert close_calls == 1
    tb = exc_info.tb
    execute_locals = None
    while tb is not None:
        if tb.tb_frame.f_code.co_name == "execute":
            execute_locals = tb.tb_frame.f_locals
            break
        tb = tb.tb_next
    assert execute_locals is not None
    assert execute_locals.get("lazyframe") is None
