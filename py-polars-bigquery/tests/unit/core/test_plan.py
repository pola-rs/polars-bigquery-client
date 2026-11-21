"""Unit tests for `polars_bigquery.core.plan` and `core.executors.base`."""

from __future__ import annotations

import dataclasses

import arrow_bigquery
import polars as pl
import pytest
from polars_bigquery.core.executors.base import PlanExecutor, apply_local_plan
from polars_bigquery.core.plan import (
    Filter,
    PlanNode,
    Project,
    QueryScan,
    Slice,
    TableScan,
    build_io_source_plan,
    decompose_linear_plan,
    validate_and_normalize_linear_plan,
)


def _table_id() -> arrow_bigquery.BigQueryTableId:
    return arrow_bigquery.BigQueryTableId("proj", "ds", "tbl")


def test_query_scan_validation_and_immutability() -> None:
    scan = QueryScan(raw_query="SELECT a, b FROM t", columns=("a", "b"))
    assert scan.children() == ()
    assert scan.maintain_order is True
    assert scan.arrow_buffer_compression == "lz4frame"
    assert scan.use_query_cache is None
    assert scan.explicit_schema is False

    with pytest.raises(dataclasses.FrozenInstanceError):
        scan.raw_query = "SELECT 2"  # type: ignore[misc]

    with pytest.raises(ValueError, match="at least one column"):
        QueryScan(raw_query="SELECT 1", columns=())

    with pytest.raises(ValueError, match="unique"):
        QueryScan(raw_query="SELECT 1", columns=("a", "b", "a"))


def test_table_scan_validation_and_mutation_isolation() -> None:
    caller_schema = pl.Schema({"a": pl.Int64(), "b": pl.String()})
    scan = TableScan(table_ref=_table_id(), schema=caller_schema)
    assert scan.children() == ()
    assert scan.columns == ("a", "b")
    assert scan.maintain_order is False

    # Mutating caller's schema after construction must not affect TableScan.schema.
    caller_schema["c"] = pl.Boolean()
    assert list(scan.schema.keys()) == ["a", "b"]
    assert scan.columns == ("a", "b")

    # TableScan holds pl.Schema (unhashable OrderedDict subclass).
    with pytest.raises(TypeError, match="unhashable"):
        hash(scan)

    with pytest.raises(ValueError, match="non-empty schema"):
        TableScan(table_ref=_table_id(), schema=pl.Schema())

    with pytest.raises(TypeError):
        TableScan(
            table_ref=_table_id(),
            schema={"a": pl.Datetime},  # type: ignore[arg-type]
        )


def test_filter_equality_and_unhashable() -> None:
    scan = QueryScan(raw_query="SELECT a FROM t", columns=("a", "b"))
    f1 = Filter(input=scan, predicate=pl.col("a") == 1)
    f2 = Filter(input=scan, predicate=pl.col("a") == 1)
    f3 = Filter(input=scan, predicate=pl.col("a") == 2)
    other_scan = QueryScan(raw_query="SELECT a FROM other", columns=("a", "b"))
    f4 = Filter(input=other_scan, predicate=pl.col("a") == 1)

    assert f1.children() == (scan,)
    assert f1 == f2
    assert f1 != f3
    assert f1 != f4
    assert (f1 == "not-a-filter") is False
    assert f1.__eq__("not-a-filter") is NotImplemented

    with pytest.raises(TypeError, match="unhashable"):
        hash(f1)


def test_slice_validation_and_children() -> None:
    scan = QueryScan(raw_query="SELECT a FROM t", columns=("a",))

    s_zero = Slice(input=scan, offset=0, length=0)
    assert s_zero.children() == (scan,)
    assert s_zero.offset == 0
    assert s_zero.length == 0

    for bad_val in (True, False, 1.5, "1", None):
        with pytest.raises(TypeError, match="Slice offset must be an int"):
            Slice(input=scan, offset=bad_val, length=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Slice length must be an int"):
            Slice(input=scan, offset=0, length=bad_val)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Slice offset must be non-negative"):
        Slice(input=scan, offset=-1, length=1)
    with pytest.raises(ValueError, match="Slice length must be non-negative"):
        Slice(input=scan, offset=0, length=-5)


def test_project_validation_and_children() -> None:
    scan = QueryScan(raw_query="SELECT a, b FROM t", columns=("a", "b"))
    empty = Project(input=scan, columns=())
    assert empty.children() == (scan,)
    assert empty.columns == ()

    with pytest.raises(ValueError, match="Project columns must be unique"):
        Project(input=scan, columns=("a", "b", "a"))


def test_decompose_linear_plan_and_unknown_node() -> None:
    scan = QueryScan(raw_query="SELECT a, b FROM t", columns=("a", "b"))
    s_leaf, ops_leaf = decompose_linear_plan(scan)
    assert s_leaf == scan
    assert ops_leaf == ()

    @dataclasses.dataclass(frozen=True, slots=True)
    class _UnknownNode(PlanNode):
        def children(self) -> tuple[PlanNode, ...]:
            return ()

    unknown = _UnknownNode()
    assert unknown.children() == ()
    with pytest.raises(TypeError, match="Unsupported plan node type: _UnknownNode"):
        decompose_linear_plan(unknown)

    wrapped = Project(input=unknown, columns=("a",))
    with pytest.raises(TypeError, match="Unsupported plan node type: _UnknownNode"):
        decompose_linear_plan(wrapped)


def test_validate_and_normalize_linear_plan_visibility_checks() -> None:
    scan = QueryScan(raw_query="SELECT a, b, c FROM t", columns=("a", "b", "c"))

    # 0-Project plan: unknown column in Filter
    with pytest.raises(ValueError, match="Filter references unknown or hidden"):
        validate_and_normalize_linear_plan(
            Filter(input=scan, predicate=pl.col("z") == 1)
        )

    # 1-Project plan: unknown column in Project
    with pytest.raises(ValueError, match="Project references unknown or hidden"):
        validate_and_normalize_linear_plan(Project(input=scan, columns=("a", "z")))

    # Multi-Project plan: Filter referencing column dropped by earlier Project
    dropped_before_filter = Filter(
        input=Project(input=scan, columns=("a", "b")),
        predicate=pl.col("c") > 0,
    )
    with pytest.raises(ValueError, match="'c'"):
        validate_and_normalize_linear_plan(dropped_before_filter)

    # Multi-Project plan: second Project referencing column dropped by first Project
    dropped_before_project = Project(
        input=Project(input=scan, columns=("a",)),
        columns=("b",),
    )
    with pytest.raises(ValueError, match="'b'"):
        validate_and_normalize_linear_plan(dropped_before_project)

    # Plan with empty Project(()) followed by Filter using column raises
    with pytest.raises(ValueError, match="'a'"):
        validate_and_normalize_linear_plan(
            Filter(input=Project(input=scan, columns=()), predicate=pl.col("a") == 1)
        )


def test_validate_and_normalize_collapses_intermediate_projects() -> None:
    scan = QueryScan(raw_query="SELECT a, b, c FROM t", columns=("a", "b", "c"))
    p1 = Filter(input=scan, predicate=pl.col("c") > 0)
    p2 = Project(input=p1, columns=("a", "b"))
    p3 = Slice(input=p2, offset=1, length=5)
    p4 = Filter(input=p3, predicate=pl.col("b") == "x")
    p5 = Project(input=p4, columns=("a",))
    p6 = Slice(input=p5, offset=0, length=2)

    source, normalized_ops, normalized_plan = validate_and_normalize_linear_plan(p6)
    assert source == scan
    # Superseded p2 Project is removed; final p5 Project stays between Filter and Slice.
    assert [type(op) for op in normalized_ops] == [
        Filter,
        Slice,
        Filter,
        Project,
        Slice,
    ]
    assert isinstance(normalized_ops[3], Project)
    assert normalized_ops[3].columns == ("a",)
    # Every surviving node is re-chained so there are no stale .input references.
    assert decompose_linear_plan(normalized_plan) == (source, normalized_ops)
    assert normalized_ops[0].input == scan
    assert normalized_ops[1].input == normalized_ops[0]
    assert normalized_ops[2].input == normalized_ops[1]
    assert normalized_ops[3].input == normalized_ops[2]
    assert normalized_ops[4].input == normalized_ops[3]


def test_validate_and_normalize_no_ops_and_table_scan() -> None:
    tscan = TableScan(
        table_ref=_table_id(),
        schema=pl.Schema({"a": pl.Int64(), "b": pl.String()}),
    )
    src, ops, norm = validate_and_normalize_linear_plan(tscan)
    assert src == tscan
    assert ops == ()
    assert norm == tscan


def test_build_io_source_plan_variants() -> None:
    scan = QueryScan(raw_query="SELECT a, b FROM t", columns=("a", "b"))

    plan_all = build_io_source_plan(
        scan, with_columns=None, predicate=None, n_rows=None
    )
    src, ops = decompose_linear_plan(plan_all)
    assert src == scan
    assert ops == (Project(input=scan, columns=("a", "b")),)

    plan_zero_limit = build_io_source_plan(
        scan, with_columns=[], predicate=pl.col("a") > 1, n_rows=0
    )
    src2, ops2 = decompose_linear_plan(plan_zero_limit)
    assert src2 == scan
    assert [type(op) for op in ops2] == [Slice, Filter, Project]
    assert ops2[0] == Slice(input=scan, offset=0, length=0)
    assert ops2[1] == Filter(input=ops2[0], predicate=pl.col("a") > 1)
    assert ops2[2] == Project(input=ops2[1], columns=())


def test_apply_local_plan_operator_orderings() -> None:
    scan = QueryScan(raw_query="SELECT a, b FROM t", columns=("a", "b"))
    base = pl.LazyFrame({"a": [1, 2, 3, 4, 5, 6], "b": ["u", "v", "w", "x", "y", "z"]})

    # Filter -> Slice(1, 2) -> Project(("b", "a"))
    filter_then_slice = Project(
        input=Slice(
            input=Filter(input=scan, predicate=pl.col("a") > 2),
            offset=1,
            length=2,
        ),
        columns=("b", "a"),
    )
    df1 = apply_local_plan(base, filter_then_slice).collect()
    assert df1.columns == ["b", "a"]
    assert df1.to_dict(as_series=False) == {"b": ["x", "y"], "a": [4, 5]}

    # Slice(0, 3) -> Filter(a > 2) -> Project(("a",))
    slice_then_filter = Project(
        input=Filter(
            input=Slice(input=scan, offset=0, length=3),
            predicate=pl.col("a") > 2,
        ),
        columns=("a",),
    )
    df2 = apply_local_plan(base, slice_then_filter).collect()
    assert df2.to_dict(as_series=False) == {"a": [3]}

    # Multi-stage Filter -> Slice -> Filter
    multi = Filter(
        input=Slice(
            input=Filter(input=scan, predicate=pl.col("a") >= 2),
            offset=1,
            length=3,
        ),
        predicate=pl.col("a") != 4,
    )
    df3 = apply_local_plan(base, multi).collect()
    assert df3.to_dict(as_series=False) == {"a": [3, 5], "b": ["w", "y"]}

    # Empty Project(()) drops all columns via .drop(pl.all())
    zero_col = Project(input=scan, columns=())
    df_empty_proj = apply_local_plan(base, zero_col).collect()
    assert df_empty_proj.width == 0
    assert df_empty_proj.columns == []


def test_plan_executor_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        PlanExecutor()  # type: ignore[abstract]


def test_query_scan_maximum_bytes_billed_validation() -> None:
    from polars_bigquery.core.plan import MAX_BIGQUERY_INT64

    assert (
        QueryScan(
            raw_query="SELECT a FROM t",
            columns=("a",),
            maximum_bytes_billed=0,
        ).maximum_bytes_billed
        == 0
    )
    assert (
        QueryScan(
            raw_query="SELECT a FROM t",
            columns=("a",),
            maximum_bytes_billed=MAX_BIGQUERY_INT64,
        ).maximum_bytes_billed
        == MAX_BIGQUERY_INT64
    )

    for bad_type in (True, False, 1.5, "100"):
        with pytest.raises(TypeError, match="maximum_bytes_billed must be an int"):
            QueryScan(
                raw_query="SELECT a FROM t",
                columns=("a",),
                maximum_bytes_billed=bad_type,  # type: ignore[arg-type]
            )

    for bad_neg in (-1, -(10**5000)):
        with pytest.raises(
            ValueError, match="maximum_bytes_billed must be non-negative"
        ):
            QueryScan(
                raw_query="SELECT a FROM t",
                columns=("a",),
                maximum_bytes_billed=bad_neg,
            )

    for bad_huge in (MAX_BIGQUERY_INT64 + 1, 10**5000):
        with pytest.raises(
            ValueError, match=f"maximum_bytes_billed must be <= {MAX_BIGQUERY_INT64}"
        ):
            QueryScan(
                raw_query="SELECT a FROM t",
                columns=("a",),
                maximum_bytes_billed=bad_huge,
            )


def test_validate_and_normalize_rejects_unexpanded_selectors() -> None:
    import polars.selectors as cs

    scan = QueryScan(raw_query="SELECT a, b FROM t", columns=("a", "b"))
    for expr in (
        pl.all_horizontal(pl.col("a", "b") == 1),
        pl.any_horizontal(pl.all() > 0),
        pl.all_horizontal(cs.integer() > 0),
        pl.col("*") == 1,
        pl.col("^a.*$") == 1,
    ):
        with pytest.raises(ValueError, match="unexpanded selector"):
            validate_and_normalize_linear_plan(Filter(input=scan, predicate=expr))


def test_apply_local_plan_rejects_oversized_slice_and_streaming_handles_it() -> None:
    from polars_bigquery.core.executors.base import execute_local_plan_batches

    scan = QueryScan(raw_query="SELECT a, b FROM t", columns=("a", "b"))
    u64_max = (1 << 64) - 1
    big_slice = Project(
        input=Filter(
            input=Slice(input=scan, offset=0, length=u64_max),
            predicate=pl.col("a") >= 2,
        ),
        columns=("b",),
    )
    base = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    with pytest.raises(ValueError, match="exceeds Polars native slice limits"):
        apply_local_plan(base, big_slice)

    _, ops = decompose_linear_plan(big_slice)
    batches = list(execute_local_plan_batches(base, ops, batch_size=1))
    assert sorted(pl.concat(batches)["b"].to_list()) == ["y", "z"]

    # Offset past total rows returns empty cleanly without OverflowError.
    huge_offset = Slice(input=scan, offset=(1 << 63) + 5, length=10)
    _, off_ops = decompose_linear_plan(huge_offset)
    assert list(execute_local_plan_batches(base, off_ops, batch_size=2)) == []

    # Length 0 after oversized offset / direct oversized zero-len slice.
    zero_after_big = Slice(
        input=Slice(input=scan, offset=0, length=u64_max),
        offset=0,
        length=0,
    )
    _, z_ops = decompose_linear_plan(zero_after_big)
    zero_batches = list(execute_local_plan_batches(base, z_ops, batch_size=1))
    assert sum(b.height for b in zero_batches) == 0


def test_iter_apply_ops_binds_chained_filters_and_projects_independently() -> None:
    """Regression test: loop-variable closure aliasing must not drop earlier Filter/Project ops."""
    from polars_bigquery.core.executors.base import (
        _iter_apply_ops,
        execute_local_plan_batches,
    )

    scan = QueryScan(raw_query="SELECT a, b FROM t", columns=("a", "b"))
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [10, 10, 10, 20]})

    chained_filters = (
        Slice(input=scan, offset=0, length=1 << 63),
        Filter(input=scan, predicate=pl.col("a") > 2),
        Filter(input=scan, predicate=pl.col("b") == 10),
    )
    out = pl.concat(list(execute_local_plan_batches(df.lazy(), chained_filters)))
    assert out.to_dict(as_series=False) == {"a": [3], "b": [10]}

    # First filter rejects every row; second filter must NOT revive them.
    empty_filters = (
        Slice(input=scan, offset=0, length=1 << 63),
        Filter(input=scan, predicate=pl.col("a") == 999),
        Filter(input=scan, predicate=pl.col("b") == 10),
    )
    assert list(execute_local_plan_batches(df.lazy(), empty_filters)) == []

    # Distinct chained Project operators must each bind their own column tuple.
    chained_projects = (
        Project(input=scan, columns=("a", "b")),
        Filter(input=scan, predicate=pl.col("a") >= 3),
        Project(input=scan, columns=()),
    )
    projected_batches = list(_iter_apply_ops(iter([df]), chained_projects))
    assert len(projected_batches) == 1
    assert projected_batches[0].shape == (2, 0)


def test_iter_apply_ops_closes_upstream_iterator_on_early_exit() -> None:
    from polars_bigquery.core.executors.base import _iter_apply_ops

    scan = QueryScan(raw_query="SELECT a FROM t", columns=("a",))

    closed_zero = False
    pulled_zero = 0

    def upstream_zero():
        nonlocal closed_zero, pulled_zero
        try:
            pulled_zero += 1
            yield pl.DataFrame({"a": [1, 2, 3]})
        finally:
            closed_zero = True

    gen0 = upstream_zero()
    assert list(_iter_apply_ops(gen0, (Slice(scan, offset=0, length=0),))) == []
    assert pulled_zero == 0
    assert (
        closed_zero is False
    )  # unstarted generator's close() completes cleanly without pulling

    # Started generator must trigger its `finally` immediately on slice stop.
    closed_limit = False

    def upstream_many():
        nonlocal closed_limit
        try:
            yield pl.DataFrame({"a": [1, 2]})
            yield pl.DataFrame({"a": [3, 4]})
            yield pl.DataFrame({"a": [5, 6]})
        finally:
            closed_limit = True

    limited = list(
        _iter_apply_ops(
            upstream_many(),
            (
                Filter(scan, predicate=pl.col("a") >= 1),
                Slice(scan, offset=0, length=2),
                Project(scan, columns=("a",)),
            ),
        )
    )
    assert pl.concat(limited)["a"].to_list() == [1, 2]
    assert closed_limit is True

    # Explicit downstream `.close()` cascades through chained filter/project generators.
    closed_downstream = False

    def upstream_cascading():
        nonlocal closed_downstream
        try:
            yield pl.DataFrame({"a": [1]})
            yield pl.DataFrame({"a": [2]})
        finally:
            closed_downstream = True

    stream = _iter_apply_ops(
        upstream_cascading(),
        (
            Filter(scan, predicate=pl.col("a") >= 1),
            Project(scan, columns=("a",)),
        ),
    )
    first = next(stream)
    assert first.to_dict(as_series=False) == {"a": [1]}
    stream.close()  # type: ignore[attr-defined]
    assert closed_downstream is True
